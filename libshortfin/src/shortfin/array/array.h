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

#include "shortfin/array/dims.h"
#include "shortfin/array/dtype.h"
#include "shortfin/array/storage.h"
#include "shortfin/array/xtensor_bridge.h"
#include "shortfin/local/program_interfaces.h"
#include "shortfin/support/api.h"

namespace shortfin::array {

// Either a host or device nd-array view.
class SHORTFIN_API base_array {
 public:
  base_array(std::span<const Dims::value_type> shape, DType dtype)
      : dtype_(dtype) {
    set_shape(shape);
  }
  // Need to explicitly define copy/move constructors even though this is
  // a value type because the Dims union is otherwise not copy/movable.
  base_array(const base_array &other)
      : base_array(other.shape(), other.dtype()) {}
  base_array(base_array &&other)
      : dtype_(other.dtype_), shape_(std::move(other.shape_)) {}
  virtual ~base_array() = default;
  virtual std::string to_s() const = 0;

  DType dtype() const { return dtype_; }

  // Access shape.
  void set_shape(std::span<const Dims::value_type> shape) { shape_.set(shape); }
  std::span<const Dims::value_type> shape() const { return shape_.span(); }
  std::span<Dims::value_type> mutable_shape() { return shape_.span(); }

  // Sometimes we need to access the raw shape container (i.e. for adapters,
  // etc).
  Dims &shape_container() { return shape_; }
  const Dims &shape_container() const { return shape_; }

 private:
  DType dtype_;
  Dims shape_;
};

class SHORTFIN_API device_array
    : public base_array,
      public poly_xt_mixin<device_array, class mapping>,
      public local::ProgramInvocationMarshalable {
 public:
  device_array(class storage storage, std::span<const Dims::value_type> shape,
               DType dtype)
      : base_array(shape, dtype), storage_(std::move(storage)) {}

  class storage &storage() { return storage_; }
  local::ScopedDevice &device() { return storage_.device(); }

  // Allocate an array on the device.
  static device_array for_device(local::ScopedDevice &device,
                                 std::span<const Dims::value_type> shape,
                                 DType dtype) {
    return device_array(
        storage::allocate_device(device, dtype.compute_dense_nd_size(shape)),
        shape, dtype);
  }

  // Allocates a host array that is registered by the device. This can include
  // arrays that are visible from different combinations of host/device.
  static device_array for_host(local::ScopedDevice &device,
                               std::span<const Dims::value_type> shape,
                               DType dtype) {
    return device_array(
        storage::allocate_host(device, dtype.compute_dense_nd_size(shape)),
        shape, dtype);
  }

  // Allocates a host array for transfer to/from this array.
  device_array for_transfer() {
    return for_host(storage().device(), shape(), dtype());
  }

  // Enqueues a fill of the storage with an arbitrary pattern of the given
  // size. The pattern size must be 1, 2, or 4. Equivalent to calling the same
  // on the backing storage.
  void fill(const void *pattern, iree_host_size_t pattern_length) {
    storage_.fill(pattern, pattern_length);
  }

  // Performs either a d2h, h2d or d2d transfer from a source storage to this
  // storage. Equivalent to calling the same on the backing storage.
  void copy_from(device_array &source_array) {
    storage_.copy_from(source_array.storage_);
  }
  // Inverse of copy_from.
  void copy_to(device_array &dest_array) {
    dest_array.storage_.copy_from(storage_);
  }

  // Untyped access to the backing data. The array must be mappable. Specific
  // access modes:
  // * data(): Read-only access to the data.
  // * data_rw(): Read/write access to the data.
  // * data_w(): Write-only access to the data with discard (initial contents
  //     are undefined.)
  const mapping data() const;
  mapping data();
  // Map the array's data for read-write untyped access.
  mapping data_rw();
  // Map the array's data for write-only untyped access.
  mapping data_w();

  // Maps memory for bridging to xtensor. If mapping is unsupported, return {}.
  std::optional<mapping> map_memory_for_xtensor();

  // Typed access to the backing data.
  template <typename EltTy>
  typed_mapping<EltTy> typed_data() {
    return typed_mapping<EltTy>(data());
  }
  template <typename EltTy>
  typed_mapping<const EltTy> typed_data() const {
    return typed_mapping<const EltTy>(data());
  }
  template <typename EltTy>
  typed_mapping<EltTy> typed_data_rw() {
    return typed_mapping<EltTy>(data_rw());
  }
  template <typename EltTy>
  typed_mapping<EltTy> typed_data_w() {
    return typed_mapping<EltTy>(data_w());
  }

  std::string to_s() const override;

 protected:
  class storage storage_;

 private:
  // ProgramInvocationMarshalable implementation.
  void AddAsInvocationArgument(local::ProgramInvocation *inv,
                               local::ProgramResourceBarrier barrier) override;
};

}  // namespace shortfin::array

#endif  // SHORTFIN_ARRAY_ARRAY_H
