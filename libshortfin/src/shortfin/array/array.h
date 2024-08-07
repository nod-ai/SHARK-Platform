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
 private:
  // Storage(LocalScope &scope, iree_hal_buffer_t *buffer)
  //     : scope_(scope), buffer_(buffer) {}
  // LocalScope &scope_;
  // iree_hal_buffer_t *buffer_;
};

}  // namespace shortfin::array

#endif  // SHORTFIN_ARRAY_ARRAY_H

// // Base class for buffers attached to the local system.
// class SHORTFIN_API LocalBuffer {
//  public:
//  protected:
//   // Initialized with a backing iree_hal_buffer_t reference count transferred
//   // to this object.
//   LocalBuffer(LocalScope &owner, iree_hal_buffer_t *buffer)
//       : owner_(owner), buffer_(buffer) {}
//   LocalScope &owner_;
//   iree_hal_buffer_t *buffer_;
// };

// // Unpeered buffer that which exists solely on the device.
// class LocalDeviceBuffer : public LocalBuffer {};
