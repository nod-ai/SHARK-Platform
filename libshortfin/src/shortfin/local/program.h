// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_PROGRAM_H
#define SHORTFIN_LOCAL_PROGRAM_H

#include <filesystem>
#include <optional>
#include <string_view>
#include <vector>

#include "shortfin/local/async.h"
#include "shortfin/local/worker.h"
#include "shortfin/support/api.h"
#include "shortfin/support/iree_helpers.h"

namespace shortfin::local {

class SHORTFIN_API Scope;
class SHORTFIN_API System;

// State related to making an invocation of a function on a program.
//
// Since ownership of this object is transferred to the loop/callback and
// internal pointers into it must remain stable, it is only valid to heap
// allocate it.
class SHORTFIN_API ProgramInvocation {
  struct Deleter {
    void operator()(ProgramInvocation *);
  };

 public:
  // The fact that we traffic in invocation pointers based on unique_ptr
  // is incidental. By cloaking its public interface this way, we use the
  // unique_ptr machinery but template meta-programming that is specialized
  // for unique_ptr sees this as a bespoke class (which is what we want because
  // ownership semantics are special).
  class Ptr : private std::unique_ptr<ProgramInvocation, Deleter> {
   public:
    using unique_ptr::unique_ptr;
    using unique_ptr::operator=;
    using unique_ptr::operator->;
    using unique_ptr::operator bool;
    using unique_ptr::get;
    using unique_ptr::release;
  };
  static_assert(sizeof(Ptr) == sizeof(void *));
  using Future = TypedFuture<ProgramInvocation::Ptr>;

  static Ptr New(std::shared_ptr<Scope> scope, iree::vm_context_ptr vm_context,
                 iree_vm_function_t &vm_function);
  ProgramInvocation(const ProgramInvocation &) = delete;
  ProgramInvocation &operator=(const ProgramInvocation &) = delete;
  ProgramInvocation &operator=(ProgramInvocation &&) = delete;
  ProgramInvocation(ProgramInvocation &&inv) = delete;
  ~ProgramInvocation();

  // Whether the ProgramInvocation has entered the scheduled state. Once
  // scheduled, arguments and initialization parameters can no longer be
  // accessed.
  bool scheduled() const { return scheduled_; }

  // Adds a ref object argument. Most of our classes which can be cast to
  // a VM object will have a type caster to produce such an instance.
  void AddArg(iree::vm_opaque_ref ref);  // Moves a reference in.
  void AddArg(iree_vm_ref_t *ref);       // Borrows the reference.

  // Transfers ownership of an invocation and schedules it on worker, returning
  // a future that will resolve to the owned invocation upon completion.
  static ProgramInvocation::Future Invoke(ProgramInvocation::Ptr invocation);

 private:
  ProgramInvocation();
  void CheckNotScheduled();

  // Parameters needed to make the async call are stored at construction time
  // up until the point the call is made in the params union. When invoking,
  // these will be copied to the stack and passed to the async invocation,
  // which initializes the async_invoke_state. Phasing it like this saves
  // hundreds of bytes of redundant storage.
  struct Params {
    // Context is retained upon construction and released when scheduled.
    iree_vm_context_t *context;
    iree_vm_function_t function;
    iree_vm_list_t *arg_list = nullptr;
  };
  union State {
    State() {}
    ~State() {}
    Params params;
    iree_vm_async_invoke_state_t async_invoke_state;
  } state;

  std::shared_ptr<Scope> scope_;
  iree_vm_list_t *result_list_ = nullptr;
  std::optional<Future> future_;
  bool scheduled_ = false;
};

// References a function in a Program.
class SHORTFIN_API ProgramFunction {
 public:
  ProgramFunction(iree::vm_context_ptr vm_context,
                  iree_vm_function_t vm_function)
      : vm_context_(std::move(vm_context)), vm_function_(vm_function) {}

  operator bool() const { return vm_context_; }

  std::string_view name() const;
  std::string_view calling_convention() const;

  ProgramInvocation::Ptr CreateInvocation(std::shared_ptr<Scope> scope);

  std::string to_s() const;

  operator iree_vm_context_t *() { return vm_context_.get(); }
  operator iree_vm_function_t &() { return vm_function_; }

 private:
  // The context that this function was resolved against.
  iree::vm_context_ptr vm_context_;
  iree_vm_function_t vm_function_;
  friend class Program;
};

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
  iree_vm_module_t *vm_module() const { return vm_module_; }
  std::string_view name() const;

  // Loads a dynamic bytecode module (VMFB) from a path on the file system.
  static ProgramModule Load(System &system, const std::filesystem::path &path,
                            bool mmap = true);

  // Gets the name of all exported functions.
  std::vector<std::string> exports() const;

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

  // Looks up a public function by fully qualified name (i.e. module.function).
  // Returns nothing if not found.
  std::optional<ProgramFunction> LookupFunction(std::string_view name);

  // Looks up a public function by fully qualified name, throwing an
  // invalid_argument exception on failure to find.
  ProgramFunction LookupRequiredFunction(std::string_view name);

  // Gets the name of all exported functions.
  std::vector<std::string> exports() const;

 private:
  explicit Program(iree::vm_context_ptr vm_context)
      : vm_context_(std::move(vm_context)) {}
  iree::vm_context_ptr vm_context_;
  friend class Scope;
};

// Implemented by a class if it can marshal itself to an invocation as an
// argument.
class ProgramInvocationMarshalable {
 public:
};

}  // namespace shortfin::local

#endif  // SHORTFIN_LOCAL_PROGRAM_H
