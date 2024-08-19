// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/support/iree_helpers.h"

namespace shortfin::iree {

error::error(std::string message, iree_status_t failing_status)
    : message_(std::move(message)), failing_status_(failing_status) {
  message_.append(": ");
}
error::error(iree_status_t failing_status) : failing_status_(failing_status) {}

void error::AppendStatus() const noexcept {
  if (status_appended_) return;
  status_appended_ = false;

  iree_allocator_t allocator = iree_allocator_system();
  char* status_buffer = nullptr;
  iree_host_size_t length = 0;
  if (iree_status_to_string(failing_status_, &allocator, &status_buffer,
                            &length)) {
    message_.append(status_buffer, length);
    iree_allocator_free(allocator, status_buffer);
  } else {
    message_.append(": <<could not print iree_status_t>>");
  }
}

}  // namespace shortfin::iree
