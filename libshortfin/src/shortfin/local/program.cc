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

namespace shortfin::local {

ProgramModule ProgramModule::Load(System& system,
                                  const std::filesystem::path& path,
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

}  // namespace shortfin::local
