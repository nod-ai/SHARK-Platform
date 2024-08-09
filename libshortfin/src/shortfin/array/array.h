// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_ARRAY_ARRAY_H
#define SHORTFIN_ARRAY_ARRAY_H

#include <algorithm>
#include <array>
#include <memory>
#include <string_view>

#include "shortfin/array/dtype.h"
#include "shortfin/array/storage.h"
#include "shortfin/support/api.h"

namespace shortfin::array {

// Either a host or device nd-array view.
class SHORTFIN_API base_array {
 public:
  base_array(std::span<const size_t> shape, DType dtype) : dtype_(dtype) {
    set_shape(shape);
  }
  // Need to explicitly define copy/move constructors even though this is
  // a value type because the Dims union is otherwise not copy/movable.
  base_array(const base_array &other)
      : base_array(other.shape(), other.dtype()) {}
  base_array(base_array &&other) : rank_(other.rank_), dtype_(other.dtype_) {
    // Custom move the dims to avoid an additional allocation. This could just
    // be a memcpy on most impls, but this is the "right way".
    if (rank_ > MAX_INLINE_RANK) {
      // Dynamic allocation.
      new (&shape_.dynamic_dims) Dims();
      shape_.dynamic_dims = std::move(other.shape_.dynamic_dims);
    } else {
      // Inline allocation.
      new (&shape_.inline_dims) Dims();
      shape_.inline_dims = other.shape_.inline_dims;
    }
    other.rank_ = 0;
  }
  virtual ~base_array() { ClearDims(); }

  DType dtype() const { return dtype_; }

  // Access shape.
  void set_shape(std::span<const size_t> shape) {
    ClearDims();
    rank_ = shape.size();
    if (rank_ > MAX_INLINE_RANK) {
      // Dynamic allocation.
      new (&shape_.dynamic_dims) std::unique_ptr<size_t[]>(new size_t[rank_]);
      std::copy(shape.begin(), shape.end(), shape_.dynamic_dims.get());
    } else {
      // Inline allocation.
      new (&shape_.inline_dims) Dims();
      std::copy(shape.begin(), shape.end(), shape_.inline_dims.begin());
    }
  }
  std::span<const size_t> shape() const {
    if (rank_ > MAX_INLINE_RANK) {
      // Dynamic allocation.
      return std::span<const size_t>(shape_.dynamic_dims.get(), rank_);
    } else {
      // Inline allocation.
      return std::span<const size_t>(&shape_.inline_dims.front(), rank_);
    }
  }
  std::span<size_t> mutable_shape() {
    if (rank_ > MAX_INLINE_RANK) {
      // Dynamic allocation.
      return std::span<size_t>(shape_.dynamic_dims.get(), rank_);
    } else {
      // Inline allocation.
      return std::span<size_t>(&shape_.inline_dims.front(), rank_);
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

  // Clears shape, setting the rank to zero and deleting any non-inline
  // dimension storage.
  void ClearDims() {
    if (rank_ > MAX_INLINE_RANK) {
      shape_.dynamic_dims.~unique_ptr();
    }
    rank_ = 0;
  }

  size_t rank_ = 0;
  DType dtype_;
  Dims shape_;
};

// View over some device allocation, modeled as a dense C-order nd array.
class SHORTFIN_API device_array final : public base_array {
 public:
  device_array(class storage storage, std::span<const size_t> shape,
               DType dtype)
      : base_array(shape, dtype), storage_(std::move(storage)) {}

  static device_array allocate(ScopedDevice &device,
                               std::span<const size_t> shape, DType dtype) {
    return device_array(
        storage::AllocateDevice(device, dtype.compute_dense_nd_size(shape)),
        shape, dtype);
  }

  class storage &storage() { return storage_; }
  ScopedDevice &device() { return storage_.device(); }
  std::string to_s() const;

 private:
  class storage storage_;
};

// View over some host allocation, registered for transfer to/from the
// device.
// These arrays can either be allocated directly or ::for_transfer with
// a corresponding device_array.
class SHORTFIN_API host_array final : public base_array {
 public:
  host_array(class storage storage, std::span<const size_t> shape, DType dtype)
      : base_array(shape, dtype), storage_(std::move(storage)) {}

  static host_array allocate(ScopedDevice &device,
                             std::span<const size_t> shape, DType dtype) {
    return host_array(
        storage::AllocateHost(device, dtype.compute_dense_nd_size(shape)),
        shape, dtype);
  }

  // Allocates a host array for transfer to/from the given device array.
  static host_array for_transfer(device_array &with_device_array) {
    return allocate(with_device_array.storage().device(),
                    with_device_array.shape(), with_device_array.dtype());
  }

  class storage &storage() { return storage_; }
  ScopedDevice &device() { return storage_.device(); }
  std::string to_s() const;

 private:
  class storage storage_;
};

}  // namespace shortfin::array

#endif  // SHORTFIN_ARRAY_ARRAY_H
