// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local/async.h"

#include "shortfin/local/worker.h"
#include "shortfin/support/logging.h"

namespace shortfin::local {

// -------------------------------------------------------------------------- //
// CompletionEvent
// -------------------------------------------------------------------------- //

CompletionEvent::CompletionEvent()
    : wait_source_(iree_wait_source_immediate()) {}

CompletionEvent::CompletionEvent(iree::shared_event::ref event)
    : wait_source_(event->await()), resource_baton_(std::move(event)) {}

CompletionEvent::CompletionEvent(iree::hal_semaphore_ptr sem, uint64_t payload)
    : wait_source_(iree_hal_semaphore_await(sem, payload)),
      resource_baton_(std::move(sem)) {}

CompletionEvent::~CompletionEvent() {}

bool CompletionEvent::is_ready() {
  iree_status_code_t status_code;
  SHORTFIN_THROW_IF_ERROR(iree_wait_source_query(wait_source_, &status_code));
  return status_code == IREE_STATUS_OK;
}

bool CompletionEvent::BlockingWait(iree_timeout_t timeout) {
  auto status = iree_wait_source_wait_one(wait_source_, timeout);
  if (iree_status_is_deadline_exceeded(status)) {
    iree_status_ignore(status);
    return false;
  }
  SHORTFIN_THROW_IF_ERROR(status);
  return true;
}

// -------------------------------------------------------------------------- //
// Future
// -------------------------------------------------------------------------- //

Future::BaseState::~BaseState() = default;

Worker *Future::GetRequiredWorker() {
  Worker *current = Worker::GetCurrent();
  if (!current) {
    throw std::logic_error(
        "Constructing a Future can only be done while running on a worker");
  }
  return current;
}

Future::~Future() {
  Release();
  state_ = nullptr;
}

void Future::Retain() const {
  iree::slim_mutex_lock_guard g(state_->lock_);
  state_->ref_count_ += 1;
}

void Future::Release() const {
  BaseState *delete_state = nullptr;
  {
    iree::slim_mutex_lock_guard g(state_->lock_);
    if (--state_->ref_count_ == 0) {
      delete_state = state_;
      state_ = nullptr;
    }
  }
  if (delete_state) {
    delete delete_state;
  }
}

void Future::IssueCallbacksWithLockHeld() {
  if (state_->callbacks_.empty()) {
    return;
  }

  Worker *current_worker = Worker::GetCurrent();
  // We manually retain going into CallLowLevel and then steal the reference to
  // the state back in the callback.
  state_->ref_count_ += 1;
  if (current_worker != state_->worker_) {
    // Cross worker signal.
    state_->worker_->CallThreadsafe([state = state_]() {
      Future self(state);  // Steal reference retained prior to CallLowLevel.
      self.HandleWorkerCallback();
    });
  } else {
    // Same worker signal.
    SHORTFIN_THROW_IF_ERROR(state_->worker_->CallLowLevel(
        &Future::RawHandleWorkerCallback, state_));
  }
}

void Future::set_failure(iree_status_t failure_status) {
  iree::slim_mutex_lock_guard g(state_->lock_);
  if (state_->done_) {
    throw std::logic_error(
        "Cannot 'set_failure' on a Future that is already done");
  }
  state_->failure_status_ = failure_status;
  state_->done_ = true;
  IssueCallbacksWithLockHeld();
}

void Future::AddCallback(FutureCallback callback) {
  iree::slim_mutex_lock_guard g(state_->lock_);
  bool was_empty = state_->callbacks_.empty();
  state_->callbacks_.push_back(std::move(callback));
  if (state_->done_ && was_empty) {
    IssueCallbacksWithLockHeld();
  }
}

iree_status_t Future::RawHandleWorkerCallback(void *state_vp, iree_loop_t loop,
                                              iree_status_t status) noexcept {
  IREE_RETURN_IF_ERROR(status);
  Future::BaseState *state = static_cast<Future::BaseState *>(state_vp);
  Future self(state);  // Steal reference retained prior to CallLowLevel.
  try {
    self.HandleWorkerCallback();
  } catch (std::exception &e) {
    return iree::exception_to_status(e);
  }
  return iree_ok_status();
}

void Future::HandleWorkerCallback() {
  // Capture callbacks in lock.
  std::vector<FutureCallback> callbacks;
  {
    iree::slim_mutex_lock_guard g(state_->lock_);
    state_->callbacks_.swap(callbacks);
  }

  // Optimize for no exceptions getting thrown. But if an exception is thrown,
  // re-enqueue the remaining callbacks and do it all again, throwing the
  // single exception once submitted. This exceptional path is not expected
  // to be fast or efficient.
  for (size_t i = 0; i < callbacks.size(); ++i) {
    FutureCallback callback = std::move(callbacks[i]);
    try {
      callback(*this);
    } catch (std::exception &e) {
      logging::error("Exception raised from Future callback. Propagating: {}",
                     e.what());
      // Acquire lock and re-submit remaining.
      {
        iree::slim_mutex_lock_guard g(state_->lock_);
        for (++i; i < callbacks.size(); ++i) {
          state_->callbacks_.push_back(std::move(callbacks[i]));
        }
        IssueCallbacksWithLockHeld();
      }
      throw;
    }
  }
}

void Future::ThrowFailureWithLockHeld() {
  if (!state_->done_) {
    throw std::logic_error("Cannot get result from Future that is not done");
  }
  SHORTFIN_THROW_IF_ERROR(state_->failure_status_.ConsumeStatus());
}

}  // namespace shortfin::local
