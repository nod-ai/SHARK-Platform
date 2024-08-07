// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_ARRAY_ARRAY_H
#define SHORTFIN_ARRAY_ARRAY_H

#include "shortfin/local_scope.h"
#include "shortfin/support/api.h"

namespace shortfin::array {

// Array storage backed by an IREE buffer of some form.
class SHORTFIN_API Storage {
 public:
  LocalScope &scope() { return scope_; }

  iree_device_size_t byte_length() {
    return iree_hal_buffer_byte_length(buffer_);
  }

 private:
  Storage(LocalScope &scope, DeviceAffinity affinity,
          iree_hal_buffer_ptr buffer)
      : buffer_(std::move(buffer)), scope_(scope), affinity_(affinity) {}
  iree_hal_buffer_ptr buffer_;
  LocalScope &scope_;
  DeviceAffinity affinity_;
};

}  // namespace shortfin::array

#endif  // SHORTFIN_ARRAY_ARRAY_H
