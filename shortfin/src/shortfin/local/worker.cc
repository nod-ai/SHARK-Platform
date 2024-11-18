// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local/worker.h"

#include "shortfin/support/logging.h"

namespace shortfin::local {

// -------------------------------------------------------------------------- //
// Globals
// -------------------------------------------------------------------------- //

namespace {

// Thread local of the current Worker.
thread_local Worker* current_thread_worker = nullptr;

}  // namespace

// -------------------------------------------------------------------------- //
// Worker::Extension
// -------------------------------------------------------------------------- //

Worker::Extension::~Extension() noexcept = default;
void Worker::Extension::OnThreadStart() noexcept {}
void Worker::Extension::OnThreadStop() noexcept {}
void Worker::Extension::OnBeforeShutdownWait() noexcept {}
void Worker::Extension::OnAfterShutdownWait() noexcept {}

// -------------------------------------------------------------------------- //
// Worker
// -------------------------------------------------------------------------- //

Worker::Worker(const Options options)
    : options_(std::move(options)),
      signal_transact_(false),
      signal_ended_(false) {
  // Set up loop.
  auto OnError = +[](void* self, iree_status_t status) {
    // TODO: FIX ME.
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
  };
  // TODO: We need a way to dynamically resize this vs having a hard limit.
  iree_loop_sync_options_t loop_options = {.max_queue_depth = 2048,
                                           .max_wait_count = 2048};
  SHORTFIN_THROW_IF_ERROR(
      iree_loop_sync_allocate(loop_options, options_.allocator, &loop_sync_));
  iree_loop_sync_scope_initialize(loop_sync_, OnError, this, &loop_scope_);
  loop_ = iree_loop_sync_scope(&loop_scope_);
}

Worker::~Worker() {
  iree_loop_sync_scope_deinitialize(&loop_scope_);
  iree_loop_sync_free(loop_sync_);
  thread_.reset();
}

Worker* Worker::GetCurrent() noexcept { return current_thread_worker; }

std::string Worker::to_s() {
  return fmt::format("<Worker '{}'>", options_.name);
}

void Worker::OnThreadStart() {
  // It is not necessary to initialize extensions since their OnThreadStart
  // is called when they are constructed (which necessarily comes after
  // this call).
  current_thread_worker = this;
  // Start extensions.
  for (auto& it : extensions_) {
    it.second->OnThreadStart();
  }
}

void Worker::OnThreadStop() {
  // Any extensions that have been created must be stopped.
  for (auto& it : extensions_) {
    it.second->OnThreadStop();
  }
  current_thread_worker = nullptr;
}

iree_status_t Worker::TransactLoop(iree_status_t signal_status) {
  if (!iree_status_is_ok(signal_status)) {
    // TODO: Handle failure.
    return signal_status;
  }

  {
    // An outside thread cannot change the state we are managing without
    // entering this critical section, so it is safe to reset the event
    // here (it is not possible for it to be spurious reset).
    iree::slim_mutex_lock_guard guard(mu_);
    signal_transact_.reset();
    if (kill_) {
      // TODO: Do we want to somehow hard abort loop in flight work (vs
      // just stopping submission of new work)?
      return iree_ok_status();
    }
    next_thunks_.swap(pending_thunks_);
  }

  // Handle all callbacks.
  for (auto& next_thunk : next_thunks_) {
    // TODO: Make thunks have to return a status, propagate, and handle
    // exceptions.
    SHORTFIN_TRACE_SCOPE_NAMED("Worker::ThreadsafeCallback");
    next_thunk();
  }
  next_thunks_.clear();
  return ScheduleExternalTransactEvent();
}

iree_status_t Worker::ScheduleExternalTransactEvent() {
  return iree_loop_wait_one(
      loop_, signal_transact_.await(), iree_infinite_timeout(),
      +[](void* self, iree_loop_t loop, iree_status_t status) {
        return static_cast<Worker*>(self)->TransactLoop(status);
      },
      this);
}

int Worker::RunOnThread() {
  auto RunLoop = [&]() -> iree_status_t {
    IREE_RETURN_IF_ERROR(ScheduleExternalTransactEvent());
    for (;;) {
      {
        iree::slim_mutex_lock_guard guard(mu_);
        if (kill_) break;
      }
      IREE_RETURN_IF_ERROR(iree_loop_drain(loop_, options_.quantum));
    }
    return iree_ok_status();
  };

  OnThreadStart();
  {
    auto loop_status = RunLoop();
    if (!iree_status_is_ok(loop_status)) {
      // TODO: Ooops. Signal the overall error handler.
      iree_status_abort(loop_status);
    }
  }
  OnThreadStop();

  signal_ended_.set();
  return 0;
}

void Worker::Start() {
  if (!options_.owned_thread) {
    throw std::logic_error("Cannot start worker when owned_thread=false");
  }
  if (thread_) {
    throw std::logic_error("Cannot start Worker multiple times");
  }

  auto EntryFunction =
      +[](void* self) { return static_cast<Worker*>(self)->RunOnThread(); };
  iree_thread_create_params_t params = {
      .name = {options_.name.data(), options_.name.size()},
      // Need to make sure that the thread can access thread_ so need to create
      // then unsuspend.
      .create_suspended = true,
  };
  SHORTFIN_THROW_IF_ERROR(iree_thread_create(
      EntryFunction, this, params, options_.allocator, thread_.for_output()));
  iree_thread_resume(thread_);
}

void Worker::Kill() {
  if (options_.owned_thread && !thread_) {
    throw std::logic_error("Cannot kill a Worker that was not started");
  }
  {
    iree::slim_mutex_lock_guard guard(mu_);
    kill_ = true;
  }
  signal_transact_.set();
}

void Worker::WaitForShutdown() {
  if (!options_.owned_thread) {
    throw std::logic_error("Cannot shutdown worker when owned_thread=false");
  }
  if (!thread_) {
    throw std::logic_error("Cannot Shutdown a Worker that was not started");
  }

  // Make sure we issue before after shutdown wait callbacks.
  struct WaitGuard {
    WaitGuard(Worker& worker) : worker(worker) {
      for (auto& it : worker.extensions_) {
        it.second->OnBeforeShutdownWait();
      }
    }
    ~WaitGuard() {
      for (auto& it : worker.extensions_) {
        it.second->OnAfterShutdownWait();
      }
    }
    Worker& worker;
  };
  WaitGuard wait_guard(*this);

  for (;;) {
    auto status = iree_wait_source_wait_one(signal_ended_.await(),
                                            iree_make_timeout_ms(5000));
    if (iree_status_is_ok(status)) {
      break;
    } else if (iree_status_is_deadline_exceeded(status)) {
      logging::warn("Still waiting for worker {} to terminate", options_.name);
    } else {
      SHORTFIN_THROW_IF_ERROR(status);
    }
  }
}

void Worker::RunOnCurrentThread() {
  if (options_.owned_thread) {
    throw std::logic_error(
        "Cannot RunOnCurrentThread if worker was configured for owned_thread");
  }
  {
    iree::slim_mutex_lock_guard guard(mu_);
    if (has_run_) {
      throw std::logic_error("Cannot RunOnCurrentThread if already finished");
    }
    has_run_ = true;
  }
  RunOnThread();
}

void Worker::CallThreadsafe(std::function<void()> callback) {
  {
    iree::slim_mutex_lock_guard guard(mu_);
    pending_thunks_.push_back(std::move(callback));
  }
  signal_transact_.set();
}

iree_status_t Worker::CallLowLevel(
    iree_status_t (*callback)(void* user_data, iree_loop_t loop,
                              iree_status_t status) noexcept,
    void* user_data, iree_loop_priority_e priority) noexcept {
  return iree_loop_call(loop_, priority, callback, user_data);
}

iree_status_t Worker::WaitUntilLowLevel(
    iree_timeout_t timeout,
    iree_status_t (*callback)(void* user_data, iree_loop_t loop,
                              iree_status_t status) noexcept,
    void* user_data) {
  return iree_loop_wait_until(loop_, timeout, callback, user_data);
}

iree_status_t Worker::WaitOneLowLevel(
    iree_wait_source_t wait_source, iree_timeout_t timeout,
    iree_status_t (*callback)(void* user_data, iree_loop_t loop,
                              iree_status_t status) noexcept,
    void* user_data) {
  return iree_loop_wait_one(loop_, wait_source, timeout, callback, user_data);
}

// Time management.
iree_time_t Worker::now() { return iree_time_now(); }
iree_time_t Worker::ConvertRelativeTimeoutToDeadlineNs(
    iree_duration_t timeout_ns) {
  return iree_relative_timeout_to_deadline_ns(timeout_ns);
}

}  // namespace shortfin::local
