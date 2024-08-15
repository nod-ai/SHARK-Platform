// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_PROCESS_H
#define SHORTFIN_LOCAL_PROCESS_H

#include <memory>

#include "shortfin/local/scope.h"
#include "shortfin/local/worker.h"
#include "shortfin/support/api.h"


namespace shortfin::local {

// Processes are the primary unit of scheduling in shortfin. They are light
// weight entities that are created on a Worker and operate in an event
// driven fashion (i.e. cps, async/await, co-routines, etc).
class SHORTFIN_API Process {
public:
  Process();

  // All processes are created as a shared_ptr.
  std::shared_ptr<Process> shared_ptr() { return shared_from_this(); }

private:
  // Pid 0 is un-launched.
  int pid_ = 0;
  std::shared_ptr<System> system_;
  Scope *scope_ = nullptr;
  Worker *worker_ = nullptr;
};

using ProcessPtr = std::shared_ptr<Process>;

} // namespace shortfin::local

#endif // SHORTFIN_LOCAL_PROCESS_H
