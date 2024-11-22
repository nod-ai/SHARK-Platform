// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_ARRAY_DTYPE_H
#define SHORTFIN_ARRAY_DTYPE_H

#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>

#include "iree/hal/buffer_view.h"
#include "shortfin/support/api.h"

namespace shortfin::array {

// Wraps an iree_hal_element_type into a DType like object.
class SHORTFIN_API DType {
 public:
#define SHORTFIN_DTYPE_HANDLE(et, ident) \
  static constexpr DType ident() { return DType(et, #ident); }
#include "shortfin/array/dtypes.inl"
#undef SHORTFIN_DTYPE_HANDLE

  constexpr operator iree_hal_element_type_t() const { return et_; }

  std::string_view name() const { return name_; }

  bool is_boolean() const {
    return iree_hal_element_numerical_type_is_boolean(et_);
  }
  bool is_integer() const {
    return iree_hal_element_numerical_type_is_integer(et_);
  }
  bool is_float() const {
    return iree_hal_element_numerical_type_is_float(et_);
  }
  bool is_complex() const {
    return iree_hal_element_numerical_type_is_complex_float(et_);
  }
  size_t bit_count() const { return iree_hal_element_bit_count(et_); }
  bool is_byte_aligned() const { return iree_hal_element_is_byte_aligned(et_); }
  size_t dense_byte_count() const {
    return iree_hal_element_dense_byte_count(et_);
  }
  bool is_integer_bitwidth(size_t bitwidth) const {
    return iree_hal_element_type_is_integer(et_, bitwidth);
  }
  uint32_t numerical_type() const {
    return iree_hal_element_numerical_type(et_);
  }

  // Computes the size in bytes required to store densely packed nd-dims.
  // This presently only supports byte aligned dtypes. In the future, when
  // supporting non byte aligned, it will require dims that do not divide the
  // sub-byte type across rows. Throws invalid_argument on any failed
  // pre-condition
  iree_device_size_t compute_dense_nd_size(std::span<const size_t> dims);

  constexpr bool operator==(const DType &other) const {
    return et_ == other.et_;
  }

  // Imports a raw iree_hal_element_type_t from the ether.
  static DType import_element_type(iree_hal_element_type_t et);

  // Asserts that the sizeof EltTy is equal to the size of this dtype.
  template <typename EltTy>
  void AssertCompatibleSize() {
    if (!is_byte_aligned() || sizeof(EltTy) != dense_byte_count()) {
      throw std::invalid_argument("Incompatible element size");
    }
  }

 private:
  constexpr DType(iree_hal_element_type_t et, std::string_view name)
      : et_(et), name_(name) {}
  iree_hal_element_type_t et_;
  std::string_view name_;
};

}  // namespace shortfin::array

#endif  // SHORTFIN_ARRAY_DTYPE_H
