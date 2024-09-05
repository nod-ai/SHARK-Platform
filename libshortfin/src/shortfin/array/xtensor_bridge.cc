// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/array/xtensor_bridge.h"

#include <sstream>

namespace shortfin::array {

namespace {

template <typename ElementTy>
class typed_xt_methods final : public poly_xt_methods {
 public:
  using xt_specific_t =
      decltype(xt::adapt(static_cast<ElementTy *>(nullptr), Dims()));
  // Our specific adaptor type must fit within the memory allocation of the
  // generic adaptor type.
  static_assert(sizeof(xt_specific_t) <= sizeof(xt_generic_t));

  xt_specific_t &adaptor() {
    return *reinterpret_cast<xt_specific_t *>(adaptor_storage);
  }

  static void concrete_inplace_new(uint8_t *inst_storage, void *array_memory,
                                   size_t array_memory_size, Dims &dims) {
    // We rely on the fact that the typed_xt_methods specialization has the
    // exact same memory layout as the base class.
    static_assert(sizeof(typed_xt_methods) == sizeof(poly_xt_methods));

    typed_xt_methods *methods =
        reinterpret_cast<typed_xt_methods *>(inst_storage);
    new (methods) typed_xt_methods();
    new (methods->adaptor_storage)
        xt_specific_t(xt::adapt(static_cast<ElementTy *>(array_memory), dims));
  }

  void inplace_destruct_this() override {
    adaptor().~xt_specific_t();
    this->~typed_xt_methods();
  }

  std::string contents_to_s() override {
    std::stringstream out;
    out << adaptor();
    return out.str();
  }
};
}  // namespace

bool poly_xt_methods::inplace_new(uint8_t *inst_storage, DType dtype,
                                  void *array_memory, size_t array_memory_size,
                                  Dims &dims) {
#define POLY_XT_CASE(et, cpp_type)                            \
  case et:                                                    \
    typed_xt_methods<cpp_type>::concrete_inplace_new(         \
        inst_storage, array_memory, array_memory_size, dims); \
    return true

  switch (static_cast<iree_hal_element_type_t>(dtype)) {
    // Hot comparisons first.
    POLY_XT_CASE(IREE_HAL_ELEMENT_TYPE_FLOAT_32, float);
    POLY_XT_CASE(IREE_HAL_ELEMENT_TYPE_INT_32, int32_t);
    POLY_XT_CASE(IREE_HAL_ELEMENT_TYPE_SINT_32, int32_t);
    POLY_XT_CASE(IREE_HAL_ELEMENT_TYPE_UINT_32, uint32_t);
    POLY_XT_CASE(IREE_HAL_ELEMENT_TYPE_INT_64, int64_t);
    POLY_XT_CASE(IREE_HAL_ELEMENT_TYPE_SINT_64, int64_t);
    POLY_XT_CASE(IREE_HAL_ELEMENT_TYPE_UINT_64, uint64_t);
    POLY_XT_CASE(IREE_HAL_ELEMENT_TYPE_INT_8, int8_t);
    POLY_XT_CASE(IREE_HAL_ELEMENT_TYPE_SINT_8, int8_t);
    POLY_XT_CASE(IREE_HAL_ELEMENT_TYPE_UINT_8, uint8_t);
    POLY_XT_CASE(IREE_HAL_ELEMENT_TYPE_INT_16, int16_t);
    POLY_XT_CASE(IREE_HAL_ELEMENT_TYPE_SINT_16, int16_t);
    POLY_XT_CASE(IREE_HAL_ELEMENT_TYPE_UINT_16, uint16_t);
    POLY_XT_CASE(IREE_HAL_ELEMENT_TYPE_FLOAT_64, double);
    POLY_XT_CASE(IREE_HAL_ELEMENT_TYPE_BOOL_8, bool);
    // TODO: float16
    // POLY_XT_CASE(IREE_HAL_ELEMENT_TYPE_FLOAT_16, TODO);
    // TODO: bfloat16
    // POLY_XT_CASE(IREE_HAL_ELEMENT_TYPE_BFLOAT_16, TODO);
    // TODO: complex64
    // POLY_XT_CASE(IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64, TODO);
    // TODO: complex128
    // POLY_XT_CASE(IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128, TODO);
  }

  return false;
}

}  // namespace shortfin::array
