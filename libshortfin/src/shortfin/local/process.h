// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_PROCESS_H
#define SHORTFIN_LOCAL_PROCESS_H

#include <memory>
#include <string>

#include "shortfin/local/async.h"
#include "shortfin/local/scope.h"
#include "shortfin/local/worker.h"
#include "shortfin/support/api.h"
#include "shortfin/support/iree_concurrency.h"

namespace shortfin::local {

namespace detail {

// Processes have a unique lifetime and also can be extended from languages
// other than C++. We therefore have a more binding friendly base class that
// can be used when the Process is aggregated in some kind of foreign
// structure and external lifetime management.
class SHORTFIN_API BaseProcess {
 public:
  BaseProcess(std::shared_ptr<Scope> scope);
  BaseProcess(const BaseProcess &) = delete;
  virtual ~BaseProcess();

  // The unique pid of this process (or zero if not launched).
  int64_t pid() const;
  std::string to_s() const;
  std::shared_ptr<Scope> &scope() { return scope_; }

  // Returns a future that can be waited on for termination.
  CompletionEvent OnTermination();

 protected:
  // Launches the process.
  void Launch();

  // Subclasses will have ScheduleOnWorker() called exactly once during
  // Launch(). The subclass must eventually call Terminate(), either
  // synchronously within this call frame or asynchronously at a future point.
  virtual void ScheduleOnWorker();

  // Called when this process has asynchronously finished.
  void Terminate();

 private:
  std::shared_ptr<Scope> scope_;

  // Process control state. Since this can be accessed by multiple threads,
  // it is protected by a lock. Most process state can only be accessed on
  // the worker thread and is unprotected.
  mutable iree::slim_mutex lock_;
  // Pid is 0 if not yet started, -1 if terminated, and a postive value if
  // running.
  int64_t pid_ = 0;

  // Must be accessed within a lock. Will be null if no one has called
  // Termination().
  iree::shared_event::ref terminated_event_;
};

}  // namespace detail

// Processes are the primary unit of scheduling in shortfin. They are light
// weight entities that are created on a Worker and operate in an event
// driven fashion (i.e. cps, async/await, co-routines, etc).
class SHORTFIN_API Process : public detail::BaseProcess {
 public:
  using BaseProcess::BaseProcess;
};

}  // namespace shortfin::local

#endif  // SHORTFIN_LOCAL_PROCESS_H
