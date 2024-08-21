// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_PROGRAM_H
#define SHORTFIN_LOCAL_PROGRAM_H

#include <filesystem>
#include <string_view>

#include "shortfin/support/api.h"
#include "shortfin/support/iree_helpers.h"

namespace shortfin::local {

class SHORTFIN_API System;

// High level API for working with program modules. Think of a module as
// a shared library in a traditional Unix system:
//
//   * Has a name and access to a certain amount of metadata.
//   * Exports functions which can be resolved and invoked.
//   * Imports functions that must be resolved by previously loaded modules.
//   * Can perform arbitrary initialization activity.
//   * Are loaded into an overall ProgramContext.
//
// Modules are thread-safe and typically loaded globally (think of them as
// files on disk, although they can be composed in multiple ways), while
// loading them into a ProgramContext causes them to be linked and made
// available for specific invocations.
//
// Under the hood, these are implemented in terms of iree_vm_module_t, which
// can be either custom, builtin or loaded from bytecode.
class SHORTFIN_API ProgramModule {
 public:
  std::string to_s() const;
  iree_vm_module_t* vm_module() const { return vm_module_; }
  std::string_view name() const;

  // Loads a dynamic bytecode module (VMFB) from a path on the file system.
  static ProgramModule Load(System& system, const std::filesystem::path& path,
                            bool mmap = true);

 protected:
  explicit ProgramModule(iree::vm_module_ptr vm_module)
      : vm_module_(std::move(vm_module)) {}

 private:
  iree::vm_module_ptr vm_module_;
};

// Programs consist of ProgramModules instantiated together and capable of
// having functions invoked on them. While it is possible to construct
// programs that do not depend on device-associated state, the dominant
// use case is for programs that are compiled to operate against the device
// HAL with a list of concrete devices. Such programs are constructed from
// a Scope.
//
// While the concurrency model for programs is technically a bit broader, the
// intended use is for them to be interacted with on a single Worker in a
// non-blocking fashion. There are many advanced ways that programs can be
// constructed to straddle devices, scopes, and workers, but that is left as
// an advanced use case.
class SHORTFIN_API Program {
 public:
  struct Options {
    // Enables program-wide execution tracing (to stderr).
    bool trace_execution = false;
  };

 private:
  explicit Program(iree::vm_context_ptr context)
      : context_(std::move(context)) {}
  iree::vm_context_ptr context_;
  friend class Scope;
};

}  // namespace shortfin::local

#endif  // SHORTFIN_LOCAL_PROGRAM_H
