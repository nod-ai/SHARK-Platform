// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local/process.h"

#include "fmt/core.h"
#include "shortfin/local/system.h"
#include "shortfin/support/logging.h"

namespace shortfin::local {

detail::BaseProcess::BaseProcess(std::shared_ptr<Scope> scope)
    : scope_(std::move(scope)) {}

detail::BaseProcess::~BaseProcess() {}

int64_t detail::BaseProcess::pid() const {
  iree::slim_mutex_lock_guard g(lock_);
  return pid_;
}

std::string detail::BaseProcess::to_s() const {
  int pid;
  {
    iree::slim_mutex_lock_guard g(lock_);
    pid = pid_;
  }

  if (pid == 0) {
    return fmt::format("Process(NOT_STARTED, worker='{}')",
                       scope_->worker().name());
  } else if (pid < 0) {
    return fmt::format("Process(TERMINATED, worker='{}')",
                       scope_->worker().name());
  } else {
    return fmt::format("Process(pid={}, worker='{}')", pid,
                       scope_->worker().name());
  }
}

void detail::BaseProcess::Launch() {
  Scope* scope = scope_.get();
  {
    iree::slim_mutex_lock_guard g(lock_);
    if (pid_ != 0) {
      throw std::logic_error("Process can only be launched a single time");
    }
    pid_ = scope->system().AllocateProcess(this);
  }

  ScheduleOnWorker();
}

void detail::BaseProcess::ScheduleOnWorker() {
  logging::info("ScheduleOnWorker()");
  Terminate();
}

void detail::BaseProcess::Terminate() {
  int deallocate_pid;
  {
    iree::slim_mutex_lock_guard g(lock_);
    deallocate_pid = pid_;
    pid_ = -1;
    if (terminated_event_) {
      terminated_event_->set();
    }
  }
  if (deallocate_pid > 0) {
    scope_->system().DeallocateProcess(deallocate_pid);
  } else {
    logging::warn("Process signalled termination multiple times (ignored)");
  }
}

CompletionEvent detail::BaseProcess::OnTermination() {
  iree::slim_mutex_lock_guard g(lock_);
  if (!terminated_event_) {
    terminated_event_ = iree::shared_event::create(pid_ < 0);
  }
  return CompletionEvent(terminated_event_);
}

}  // namespace shortfin::local
