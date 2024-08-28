// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local/program.h"

#include "fmt/core.h"
#include "fmt/std.h"
#include "iree/vm/bytecode/module.h"
#include "shortfin/local/system.h"
#include "shortfin/support/logging.h"

namespace shortfin::local {

namespace {
void GetVmModuleExports(iree_vm_module_t *vm_module,
                        std::vector<std::string> &exports) {
  auto sig = iree_vm_module_signature(vm_module);
  for (iree_host_size_t i = 0; i < sig.export_function_count; ++i) {
    iree_vm_function_t f;
    SHORTFIN_THROW_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
        vm_module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &f));
    exports.emplace_back(to_string_view(iree_vm_function_name(&f)));
  }
}
}  // namespace

// -------------------------------------------------------------------------- //
// ProgramFunction
// -------------------------------------------------------------------------- //

std::string_view ProgramFunction::name() const {
  if (!*this) return {};
  return to_string_view(iree_vm_function_name(&vm_function_));
}

std::string_view ProgramFunction::calling_convention() const {
  if (!*this) return {};
  return to_string_view(
      iree_vm_function_signature(&vm_function_).calling_convention);
}

std::string ProgramFunction::to_s() const {
  if (!*this) return std::string("ProgramFunction(NULL)");
  return fmt::format("ProgramFunction({}: {})", name(), calling_convention());
}

// -------------------------------------------------------------------------- //
// ProgramModule
// -------------------------------------------------------------------------- //

ProgramModule ProgramModule::Load(System &system,
                                  const std::filesystem::path &path,
                                  bool mmap) {
  iree::file_contents_ptr contents;
  iree_file_read_flags_t flags =
      mmap ? IREE_FILE_READ_FLAG_MMAP : IREE_FILE_READ_FLAG_PRELOAD;
  SHORTFIN_THROW_IF_ERROR(iree_file_read_contents(
      path.c_str(), flags, system.host_allocator(), contents.for_output()));

  // Ownership hazard: iree_vm_bytecode_module_create only assumes ownership
  // of the contents when it returns *sucessfully*. In the exceptional case,
  // ownership remains with the caller, so we let the RAII wrapper hold on to
  // it until after success.
  iree::vm_module_ptr module;
  SHORTFIN_THROW_IF_ERROR(iree_vm_bytecode_module_create(
      system.vm_instance(), contents.const_buffer(), contents.deallocator(),
      system.host_allocator(), module.for_output()));
  contents.release();  // Must be invoked on success path only.
  return ProgramModule(std::move(module));
}

std::string_view ProgramModule::name() const {
  return to_string_view(iree_vm_module_name(vm_module_));
}

std::string ProgramModule::to_s() const {
  auto sig = iree_vm_module_signature(vm_module_);
  std::vector<std::string> exports;
  for (iree_host_size_t i = 0; i < sig.export_function_count; ++i) {
    iree_vm_function_t f;
    SHORTFIN_THROW_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
        vm_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &f));
    exports.push_back(fmt::format(
        "{}({})", to_string_view(iree_vm_function_name(&f)),
        to_string_view(iree_vm_function_signature(&f).calling_convention)));
  }
  return fmt::format("ProgramModule('{}', version={}, exports=[{}])", name(),
                     sig.version, fmt::join(exports, ", "));
}

std::vector<std::string> ProgramModule::exports() const {
  std::vector<std::string> exports;
  GetVmModuleExports(vm_module_, exports);
  return exports;
}

// -------------------------------------------------------------------------- //
// Program
// -------------------------------------------------------------------------- //

std::optional<ProgramFunction> Program::LookupFunction(std::string_view name) {
  iree_vm_function_t f;
  iree_status_t status = iree_vm_context_resolve_function(
      vm_context_, to_iree_string_view(name), &f);
  if (iree_status_is_not_found(status)) return {};
  SHORTFIN_THROW_IF_ERROR(status);
  return ProgramFunction(vm_context_, f);
}

ProgramFunction Program::LookupRequiredFunction(std::string_view name) {
  auto f = LookupFunction(name);
  if (!f) {
    throw std::invalid_argument(
        fmt::format("Function '{}' not found in program. Available exports: {}",
                    name, fmt::join(exports(), ", ")));
  }
  return std::move(*f);
}

std::vector<std::string> Program::exports() const {
  std::vector<std::string> results;

  // Iterate in reverse since "user modules" are typically last.
  int module_count = iree_vm_context_module_count(vm_context_);
  for (int i = module_count - 1; i >= 0; --i) {
    auto vm_module = iree_vm_context_module_at(vm_context_, i);
    std::string_view module_name =
        to_string_view(iree_vm_module_name(vm_module));
    std::vector<std::string> names;
    GetVmModuleExports(vm_module, names);
    for (auto &name : names) {
      results.push_back(fmt::format("{}.{}", module_name, name));
    }
  }
  return results;
}

// -------------------------------------------------------------------------- //
// Invocation
// -------------------------------------------------------------------------- //

void Invocation::Deleter::operator()(Invocation *inst) {
  uint8_t *memory = static_cast<uint8_t *>(static_cast<void *>(inst));

  // Trailing arg list and result list. The arg list pointer is only available
  // at construction, so we use the knowledge that it is stored right after
  // the object. The result_list_ is available for the life of the invocation.
  iree_vm_list_deinitialize(static_cast<iree_vm_list_t *>(
      static_cast<void *>(memory + sizeof(Invocation))));
  iree_vm_list_deinitialize(inst->result_list_);

  // Was allocated in New as a uint8_t[] so delete it by whence it came.
  delete[] memory;
}

Invocation::Invocation() = default;
Invocation::~Invocation() {
  if (!scheduled()) {
    // This instance was dropped on the floor before scheduling.
    // Clean up the initialization parameters.
    iree::vm_context_ptr drop =
        iree::vm_context_ptr::steal_reference(state.params.context);
  }
}

Invocation::Ptr Invocation::New(iree::vm_context_ptr vm_context,
                                iree_vm_function_t &vm_function) {
  auto sig = iree_vm_function_signature(&vm_function);
  iree_host_size_t arg_count;
  iree_host_size_t result_count;
  SHORTFIN_THROW_IF_ERROR(iree_vm_function_call_count_arguments_and_results(
      &sig, &arg_count, &result_count));

  // Compute size of trailing arg/result storage.
  auto variant_type_def = iree_vm_make_undefined_type_def();
  iree_host_size_t arg_storage_size =
      iree_vm_list_storage_size(&variant_type_def, arg_count);
  iree_host_size_t result_storage_size =
      iree_vm_list_storage_size(&variant_type_def, result_count);

  // Allocate storage for the Invocation, arg, result list and placement new
  // the Invocation into the storage area.
  std::unique_ptr<uint8_t[]> inst_storage(
      new uint8_t[sizeof(Invocation) + arg_storage_size + result_storage_size]);
  new (inst_storage.get()) Invocation();

  // Initialize trailing lists. Abort on failure since this is a bug and we
  // would otherwise leak.
  iree_vm_list_t *arg_list;
  iree_vm_list_t *result_list;
  IREE_CHECK_OK(
      iree_vm_list_initialize({.data = inst_storage.get() + sizeof(Invocation),
                               .data_length = arg_storage_size},
                              &variant_type_def, arg_count, &arg_list));
  IREE_CHECK_OK(iree_vm_list_initialize(
      {.data = inst_storage.get() + sizeof(Invocation) + arg_storage_size,
       .data_length = result_storage_size},
      &variant_type_def, arg_count, &result_list));

  Ptr inst(
      static_cast<Invocation *>(static_cast<void *>(inst_storage.release())),
      Deleter());
  inst->state.params.context =
      vm_context.release();  // Ref transfer to Invocation.
  inst->state.params.function = vm_function;
  inst->state.params.arg_list = arg_list;
  inst->result_list_ = result_list;
  return inst;
}

}  // namespace shortfin::local
