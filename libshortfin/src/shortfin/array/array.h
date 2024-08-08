// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_ARRAY_ARRAY_H
#define SHORTFIN_ARRAY_ARRAY_H

#include <array>
#include <memory>
#include <string_view>

#include "shortfin/array/dtype.h"
#include "shortfin/array/storage.h"

namespace shortfin::array {

// Either a host or device nd-array view.
class base_array {
 public:
  base_array(std::span<const size_t> dims, DType dtype) : dtype_(dtype) {
    set_dims(dims);
  }
  virtual ~base_array() { ClearDims(); }

  DType dtype() const { return dtype_; }

  // Access dims.
  void set_dims(std::span<const size_t> dims) {
    ClearDims();
    rank_ = dims.size();
    if (rank_ > MAX_INLINE_RANK) {
      // Dynamic allocation.
      new (&dims_.dynamic_dims) std::unique_ptr<size_t[]>(new size_t[rank_]);
      std::copy(dims.begin(), dims.end(), dims_.dynamic_dims.get());
    } else {
      // Inline allocation.
      new (&dims_.inline_dims) Dims();
      std::copy(dims.begin(), dims.end(), dims_.inline_dims.begin());
    }
  }
  std::span<const size_t> dims() const {
    if (rank_ > MAX_INLINE_RANK) {
      // Dynamic allocation.
      return std::span<const size_t>(dims_.dynamic_dims.get(), rank_);
    } else {
      // Inline allocation.
      return std::span<const size_t>(&dims_.inline_dims.front(), rank_);
    }
  }
  std::span<size_t> mutable_dims() {
    if (rank_ > MAX_INLINE_RANK) {
      // Dynamic allocation.
      return std::span<size_t>(dims_.dynamic_dims.get(), rank_);
    } else {
      // Inline allocation.
      return std::span<size_t>(&dims_.inline_dims.front(), rank_);
    }
  }

 private:
  static constexpr size_t MAX_INLINE_RANK = 6;
  union Dims {
    Dims() {}
    ~Dims() {}
    std::array<size_t, MAX_INLINE_RANK> inline_dims;
    std::unique_ptr<size_t[]> dynamic_dims;
  };

  // Clears dims, setting the rank to zero and deleting any non-inline dimension
  // storage.
  void ClearDims() {
    if (rank_ > MAX_INLINE_RANK) {
      dims_.dynamic_dims.~unique_ptr();
    }
    rank_ = 0;
  }

  size_t rank_ = 0;
  DType dtype_;
  Dims dims_;
};

// View over some device allocation, modeled as a dense C-order nd array.
class device_array final : public base_array {
 public:
  device_array(storage device_storage, std::span<const size_t> dims,
               DType dtype)
      : base_array(dims, dtype), device_storage_(std::move(device_storage)) {}

  // static device_array allocate(LocalScope &scope, std::span<const size_t>
  // dims,
  //                              DType dtype) {
  //   return device_array(storage::AllocateDevice(
  //                           scope, device,
  //                           dtype.compute_dense_nd_size(dims)),
  //                       dims, dtype);
  // }

 private:
  storage device_storage_;
};

}  // namespace shortfin::array

#endif  // SHORTFIN_ARRAY_ARRAY_H
