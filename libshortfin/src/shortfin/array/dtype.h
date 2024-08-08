// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_ARRAY_DTYPE_H
#define SHORTFIN_ARRAY_DTYPE_H

#include <optional>
#include <span>
#include <string_view>

#include "iree/hal/buffer_view.h"

namespace shortfin::array {

// Wraps an iree_hal_element_type into a DType like object.
class DType {
 public:
  static DType opaque8() {
    return DType(IREE_HAL_ELEMENT_TYPE_OPAQUE_8, "opaque8");
  }
  static DType opaque16() {
    return DType(IREE_HAL_ELEMENT_TYPE_OPAQUE_16, "opaque16");
  }
  static DType opaque32() {
    return DType(IREE_HAL_ELEMENT_TYPE_OPAQUE_32, "opaque32");
  }
  static DType opaque64() {
    return DType(IREE_HAL_ELEMENT_TYPE_OPAQUE_64, "opaque64");
  }
  static DType bool8() { return DType(IREE_HAL_ELEMENT_TYPE_BOOL_8, "bool8"); }
  static DType int4() { return DType(IREE_HAL_ELEMENT_TYPE_INT_4, "int4"); }
  static DType sint4() { return DType(IREE_HAL_ELEMENT_TYPE_SINT_4, "sint4"); }
  static DType uint4() { return DType(IREE_HAL_ELEMENT_TYPE_UINT_4, "uint4"); }
  static DType int8() { return DType(IREE_HAL_ELEMENT_TYPE_INT_8, "int8"); }
  static DType sint8() { return DType(IREE_HAL_ELEMENT_TYPE_SINT_8, "sint8"); }
  static DType uint8() { return DType(IREE_HAL_ELEMENT_TYPE_UINT_8, "uint8"); }
  static DType int16() { return DType(IREE_HAL_ELEMENT_TYPE_INT_16, "int16"); }
  static DType sint16() {
    return DType(IREE_HAL_ELEMENT_TYPE_SINT_16, "sint16");
  }
  static DType uint16() {
    return DType(IREE_HAL_ELEMENT_TYPE_UINT_16, "uint16");
  }
  static DType int32() { return DType(IREE_HAL_ELEMENT_TYPE_INT_32, "int32"); }
  static DType sint32() {
    return DType(IREE_HAL_ELEMENT_TYPE_SINT_32, "sint32");
  }
  static DType uint32() {
    return DType(IREE_HAL_ELEMENT_TYPE_UINT_32, "uint32");
  }
  static DType int64() { return DType(IREE_HAL_ELEMENT_TYPE_INT_64, "int64"); }
  static DType sint64() {
    return DType(IREE_HAL_ELEMENT_TYPE_SINT_64, "sint64");
  }
  static DType uint64() {
    return DType(IREE_HAL_ELEMENT_TYPE_UINT_64, "uint64");
  }
  static DType float16() {
    return DType(IREE_HAL_ELEMENT_TYPE_FLOAT_16, "float16");
  }
  static DType float32() {
    return DType(IREE_HAL_ELEMENT_TYPE_FLOAT_32, "float32");
  }
  static DType float64() {
    return DType(IREE_HAL_ELEMENT_TYPE_FLOAT_64, "float64");
  }
  static DType bfloat16() {
    return DType(IREE_HAL_ELEMENT_TYPE_BFLOAT_16, "bfloat16");
  }
  static DType complex64() {
    return DType(IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64, "complex64");
  }
  static DType complex128() {
    return DType(IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128, "complex128");
  }

  operator iree_hal_element_type_t() const { return et_; }

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

  // Computes the size in bytes required to store densely packed nd-dims.
  // This presently only supports byte aligned dtypes. In the future, when
  // supporting non byte aligned, it will require dims that do not divide the
  // sub-byte type across rows. Throws invalid_argument on any failed
  // pre-condition
  iree_device_size_t compute_dense_nd_size(std::span<const size_t> dims);

 private:
  DType(iree_hal_element_type_t et, std::string_view name)
      : et_(et), name_(name) {}
  iree_hal_element_type_t et_;
  std::string_view name_;
};

}  // namespace shortfin::array

#endif  // SHORTFIN_ARRAY_DTYPE_H
