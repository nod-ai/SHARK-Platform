// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./lib_ext.h"
#include "./utils.h"
#include "shortfin/array/api.h"
#include "shortfin/support/logging.h"
#include "xtensor/xsort.hpp"
#include "xtl/xhalf_float.hpp"

using namespace shortfin::array;

namespace shortfin::python {

namespace {

static const char DOCSTRING_ARGMAX[] =
    R"(Returns the indices of the maximum values along an axis.

See documentation for numpy.argmax which this operator emulates.

Implemented dtypes: float16, float32.
)";

}  // namespace

#define SF_UNARY_COMPUTE_CASE(dtype_name, cpp_type) \
  case DType::dtype_name():                         \
    return compute.template operator()<cpp_type>()

void BindArrayHostOps(py::module_ &m) {
  m.def(
      "argmax",
      [](device_array &input, int axis, std::optional<device_array> out,
         bool keepdims, bool device_visible) {
        if (axis < 0) axis += input.shape().size();
        if (axis < 0 || axis >= input.shape().size()) {
          throw std::invalid_argument(
              fmt::format("Axis out of range: Must be [0, {}) but got {}",
                          input.shape().size(), axis));
        }
        if (out && (out->dtype() != DType::int64())) {
          throw std::invalid_argument("out array must have dtype=int64");
        }
        auto compute = [&]<typename EltTy>() {
          auto input_t = input.map_xtensor<EltTy>();
          auto result = xt::argmax(*input_t, axis);
          if (!out) {
            out.emplace(device_array::for_host(input.device(), result.shape(),
                                               DType::int64(), device_visible));
          }
          auto out_t = out->map_xtensor_w<int64_t>();
          *out_t = result;
          if (keepdims) {
            out->expand_dims(axis);
          }
          return *out;
        };

        switch (input.dtype()) {
          SF_UNARY_COMPUTE_CASE(float16, half_float::half);
          SF_UNARY_COMPUTE_CASE(float32, float);
          default:
            throw std::invalid_argument(
                fmt::format("Unsupported dtype({}) for operator argmax",
                            input.dtype().name()));
        }
      },
      py::arg("input"), py::arg("axis") = -1, py::arg("out") = py::none(),
      py::kw_only(), py::arg("keepdims") = false,
      py::arg("device_visible") = false, DOCSTRING_ARGMAX);
}

}  // namespace shortfin::python
