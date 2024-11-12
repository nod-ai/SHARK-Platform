// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./lib_ext.h"
#include "./utils.h"
#include "shortfin/array/api.h"
#include "shortfin/support/logging.h"
#include "xtensor/xrandom.hpp"
#include "xtensor/xsort.hpp"
#include "xtl/xhalf_float.hpp"

using namespace shortfin::array;

namespace shortfin::python {

namespace {

static const char DOCSTRING_ARGMAX[] =
    R"(Returns the indices of the maximum values along an axis.

Implemented for dtypes: float16, float32.

Args:
  input: An input array.
  axis: Axis along which to sort. Defaults to the last axis (note that the
    numpy default is into the flattened array, which we do not support).
  keepdims: Whether to preserve the sort axis. If true, this will become a unit
    dim. If false, it will be removed.
  out: Array to write into. If specified, it must have an expected shape and
    int64 dtype.
  device_visible: Whether to make the result array visible to devices. Defaults to
    False.

Returns:
  A device_array of dtype=int64, allocated on the host and not visible to the device.
)";

static const char DOCSTRING_CONVERT[] =
    R"(Does an elementwise conversion from one dtype to another.

The same behavior exists for several conversion ops:

* `convert` : element-wise conversion like a static cast.
* `round` : element-wise nearest integer to the input, rounding halfway cases
  away from zero.
* `ceil` : element-wise smallest integer value not less than the input.
* `floor` : element-wise smallest integer value not greater than the input.
* `trunc` : element-wise nearest integer not greater in magnitude than the input.

For nearest-integer conversions (round, ceil, floor, trunc), the input dtype
must be a floating point array, and the output must be a byte-aligned integer
type between 8 and 32 bits.

Args:
  input: An input array of a floating point dtype.
  dtype: If given, then this is the explicit output dtype.
  out: If given, then the results are written to this array. This implies the
    output dtype.
  device_visible: Whether to make the result array visible to devices. Defaults to
    False.

Returns:
  A device_array of the requested dtype, or the input dtype if not specified.
)";

static const char DOCSTRING_FILL_RANDN[] =
    R"(Fills an array with numbers sampled from the standard ormal distribution.

Values are samples with a mean of 0 and standard deviation of 1.

This operates like torch.randn but only supports in place fills to an existing
array, deriving shape and dtype from the output array.

Args:
  out: Output array to fill.
  generator: Uses an explicit generator. If not specified, uses a global
    default.
)";

static const char DOCSTRING_RANDOM_GENERATOR[] =
    R"(Returns an object for generating random numbers.

  Every instance is self contained and does not share state with others.

  Args:
    seed: Optional seed for the generator. Not setting a seed will cause an
      implementation defined value to be used, which may in fact be a completely
      fixed number.
  )";

#define SF_UNARY_FUNCTION_CASE(dtype_name, cpp_type) \
  case DType::dtype_name():                          \
    return compute.template operator()<cpp_type>()

#define SF_UNARY_THUNK_CASE(dtype_name, cpp_type) \
  case DType::dtype_name():                       \
    compute.template operator()<cpp_type>();      \
    break

struct PyRandomGenerator {
 public:
  using SeedType = xt::random::default_engine_type::result_type;
  PyRandomGenerator(std::optional<SeedType> seed) {
    if (seed) SetSeed(*seed);
  }

  static PyRandomGenerator &get_default() {
    static PyRandomGenerator default_generator(std::nullopt);
    return default_generator;
  }

  void SetSeed(SeedType seed) { engine().seed(seed); }

  xt::random::default_engine_type &engine() { return engine_; }

 private:
  xt::random::default_engine_type engine_;
};

// Generic conversion templates, split into a bindable template and functors
// that operate on pre-allocated outputs.
template <typename ConvertFunc>
device_array GenericElementwiseConvert(device_array &input,
                                       std::optional<DType> dtype,
                                       std::optional<device_array> out,
                                       bool device_visible) {
  // Argument check and output allocation.
  if (!dtype) {
    dtype = out ? out->dtype() : input.dtype();
  } else {
    if (out && out->dtype() != dtype) {
      throw std::invalid_argument(
          "if both dtype and out are specified, they must match");
    }
  }
  if (!out) {
    out.emplace(device_array::for_host(input.device(), input.shape(), *dtype,
                                       device_visible));
  }

  ConvertFunc::Invoke(input, *dtype, *out);
  return *out;
}

// Generic elementwise conversion functor
struct ConvertFunctor {
  static void Invoke(device_array &input, DType dtype, device_array &out) {
    SHORTFIN_TRACE_SCOPE_NAMED("PyHostOp::convert");
    auto compute = [&]<typename EltTy>() -> void {
      auto input_t = input.map_xtensor<EltTy>();
      // Casted output.
#define SF_STORE_CASE(dtype_name, cpp_type)     \
  case DType::dtype_name(): {                   \
    auto out_t = out.map_xtensor_w<cpp_type>(); \
    *out_t = xt::cast<cpp_type>(*input_t);      \
    break;                                      \
  }
      switch (dtype) {
        SF_STORE_CASE(float16, half_float::half);
        SF_STORE_CASE(float32, float);
        SF_STORE_CASE(float64, double);
        SF_STORE_CASE(uint8, uint8_t);
        SF_STORE_CASE(int8, int8_t);
        SF_STORE_CASE(uint16, uint16_t);
        SF_STORE_CASE(int16, int16_t);
        SF_STORE_CASE(uint32, uint32_t);
        SF_STORE_CASE(int32, int32_t);
        SF_STORE_CASE(uint64, uint64_t);
        SF_STORE_CASE(int64, int64_t);
        default:
          throw std::invalid_argument("Invalid output dtype for convert op");
      }

#undef SF_STORE_CASE
    };

    switch (input.dtype()) {
      SF_UNARY_THUNK_CASE(float16, half_float::half);
      SF_UNARY_THUNK_CASE(float32, float);
      SF_UNARY_THUNK_CASE(float64, double);
      SF_UNARY_THUNK_CASE(uint8, uint8_t);
      SF_UNARY_THUNK_CASE(int8, int8_t);
      SF_UNARY_THUNK_CASE(uint16, uint16_t);
      SF_UNARY_THUNK_CASE(int16, int16_t);
      SF_UNARY_THUNK_CASE(uint32, uint32_t);
      SF_UNARY_THUNK_CASE(int32, uint32_t);
      SF_UNARY_THUNK_CASE(uint64, uint64_t);
      SF_UNARY_THUNK_CASE(int64, int64_t);
      default:
        throw std::invalid_argument(fmt::format(
            "Unsupported dtype({}) for converting nearest integer op",
            dtype.name()));
    }
  }
};

// Converting round functor.
struct ConvertRoundFunctor {
  static void Invoke(device_array &input, DType dtype, device_array &out) {
    SHORTFIN_TRACE_SCOPE_NAMED("PyHostOp::round");
    auto compute = [&]<typename EltTy>() -> void {
      auto input_t = input.map_xtensor<EltTy>();
      auto rounded = xt::round(*input_t);
      if (input.dtype() == dtype) {
        // Same type output.
        auto out_t = out.map_xtensor_w<EltTy>();
        *out_t = rounded;
      } else {
        // Casted output.
#define SF_STORE_CASE(dtype_name, cpp_type)     \
  case DType::dtype_name(): {                   \
    auto out_t = out.map_xtensor_w<cpp_type>(); \
    *out_t = xt::cast<cpp_type>(rounded);       \
    break;                                      \
  }
        switch (dtype) {
          SF_STORE_CASE(uint8, uint8_t);
          SF_STORE_CASE(int8, int8_t);
          SF_STORE_CASE(uint16, uint16_t);
          SF_STORE_CASE(int16, int16_t);
          SF_STORE_CASE(uint32, uint32_t);
          SF_STORE_CASE(int32, int32_t);
          default:
            throw std::invalid_argument(
                "Invalid output dtype for converting nearest integer op");
        }
      }
#undef SF_STORE_CASE
    };

    switch (input.dtype()) {
      SF_UNARY_THUNK_CASE(float16, half_float::half);
      SF_UNARY_THUNK_CASE(float32, float);
      default:
        throw std::invalid_argument(fmt::format(
            "Unsupported dtype({}) for converting nearest integer op",
            dtype.name()));
    }
  }
};

struct ConvertCeilFunctor {
  static void Invoke(device_array &input, DType dtype, device_array &out) {
    SHORTFIN_TRACE_SCOPE_NAMED("PyHostOp::ceil");
    auto compute = [&]<typename EltTy>() -> void {
      auto input_t = input.map_xtensor<EltTy>();
      auto rounded = xt::ceil(*input_t);
      if (input.dtype() == dtype) {
        // Same type output.
        auto out_t = out.map_xtensor_w<EltTy>();
        *out_t = rounded;
      } else {
        // Casted output.
#define SF_STORE_CASE(dtype_name, cpp_type)     \
  case DType::dtype_name(): {                   \
    auto out_t = out.map_xtensor_w<cpp_type>(); \
    *out_t = xt::cast<cpp_type>(rounded);       \
    break;                                      \
  }
        switch (dtype) {
          SF_STORE_CASE(uint8, uint8_t);
          SF_STORE_CASE(int8, int8_t);
          SF_STORE_CASE(uint16, uint16_t);
          SF_STORE_CASE(int16, int16_t);
          SF_STORE_CASE(uint32, uint32_t);
          SF_STORE_CASE(int32, int32_t);
          default:
            throw std::invalid_argument(
                "Invalid output dtype for converting nearest integer op");
        }
      }
#undef SF_STORE_CASE
    };

    switch (input.dtype()) {
      SF_UNARY_THUNK_CASE(float16, half_float::half);
      SF_UNARY_THUNK_CASE(float32, float);
      default:
        throw std::invalid_argument(fmt::format(
            "Unsupported dtype({}) for converting nearest integer op",
            dtype.name()));
    }
  }
};

struct ConvertFloorFunctor {
  static void Invoke(device_array &input, DType dtype, device_array &out) {
    SHORTFIN_TRACE_SCOPE_NAMED("PyHostOp::floor");
    auto compute = [&]<typename EltTy>() -> void {
      auto input_t = input.map_xtensor<EltTy>();
      auto rounded = xt::floor(*input_t);
      if (input.dtype() == dtype) {
        // Same type output.
        auto out_t = out.map_xtensor_w<EltTy>();
        *out_t = rounded;
      } else {
        // Casted output.
#define SF_STORE_CASE(dtype_name, cpp_type)     \
  case DType::dtype_name(): {                   \
    auto out_t = out.map_xtensor_w<cpp_type>(); \
    *out_t = xt::cast<cpp_type>(rounded);       \
    break;                                      \
  }
        switch (dtype) {
          SF_STORE_CASE(uint8, uint8_t);
          SF_STORE_CASE(int8, int8_t);
          SF_STORE_CASE(uint16, uint16_t);
          SF_STORE_CASE(int16, int16_t);
          SF_STORE_CASE(uint32, uint32_t);
          SF_STORE_CASE(int32, int32_t);
          default:
            throw std::invalid_argument(
                "Invalid output dtype for converting nearest integer op");
        }
      }
#undef SF_STORE_CASE
    };

    switch (input.dtype()) {
      SF_UNARY_THUNK_CASE(float16, half_float::half);
      SF_UNARY_THUNK_CASE(float32, float);
      default:
        throw std::invalid_argument(fmt::format(
            "Unsupported dtype({}) for converting nearest integer op",
            dtype.name()));
    }
  }
};

struct ConvertTruncFunctor {
  static void Invoke(device_array &input, DType dtype, device_array &out) {
    SHORTFIN_TRACE_SCOPE_NAMED("PyHostOp::trunc");
    auto compute = [&]<typename EltTy>() -> void {
      auto input_t = input.map_xtensor<EltTy>();
      auto rounded = xt::trunc(*input_t);
      if (input.dtype() == dtype) {
        // Same type output.
        auto out_t = out.map_xtensor_w<EltTy>();
        *out_t = rounded;
      } else {
        // Casted output.
#define SF_STORE_CASE(dtype_name, cpp_type)     \
  case DType::dtype_name(): {                   \
    auto out_t = out.map_xtensor_w<cpp_type>(); \
    *out_t = xt::cast<cpp_type>(rounded);       \
    break;                                      \
  }
        switch (dtype) {
          SF_STORE_CASE(uint8, uint8_t);
          SF_STORE_CASE(int8, int8_t);
          SF_STORE_CASE(uint16, uint16_t);
          SF_STORE_CASE(int16, int16_t);
          SF_STORE_CASE(uint32, uint32_t);
          SF_STORE_CASE(int32, int32_t);
          default:
            throw std::invalid_argument(
                "Invalid output dtype for converting nearest integer op");
        }
      }
#undef SF_STORE_CASE
    };

    switch (input.dtype()) {
      SF_UNARY_THUNK_CASE(float16, half_float::half);
      SF_UNARY_THUNK_CASE(float32, float);
      default:
        throw std::invalid_argument(fmt::format(
            "Unsupported dtype({}) for converting nearest integer op",
            dtype.name()));
    }
  }
};

}  // namespace

void BindArrayHostOps(py::module_ &m) {
  // Simple op definitions.
  m.def(
      "argmax",
      [](device_array &input, int axis, std::optional<device_array> out,
         bool keepdims, bool device_visible) {
        SHORTFIN_TRACE_SCOPE_NAMED("PyHostOp::argmax");
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
          SF_UNARY_FUNCTION_CASE(float16, half_float::half);
          SF_UNARY_FUNCTION_CASE(float32, float);
          default:
            throw std::invalid_argument(
                fmt::format("Unsupported dtype({}) for operator argmax",
                            input.dtype().name()));
        }
      },
      py::arg("input"), py::arg("axis") = -1, py::arg("out") = py::none(),
      py::kw_only(), py::arg("keepdims") = false,
      py::arg("device_visible") = false, DOCSTRING_ARGMAX);

  // Random number generation.
  py::class_<PyRandomGenerator>(m, "RandomGenerator")
      .def(py::init<std::optional<PyRandomGenerator::SeedType>>(),
           py::arg("seed") = py::none(), DOCSTRING_RANDOM_GENERATOR);
  m.def(
      "fill_randn",
      [](device_array out, std::optional<PyRandomGenerator *> gen) {
        SHORTFIN_TRACE_SCOPE_NAMED("PyHostOp::fill_randn");
        if (!gen) gen = &PyRandomGenerator::get_default();
        auto compute = [&]<typename EltTy>() {
          auto result = xt::random::randn(out.shape_container(), /*mean=*/0.0,
                                          /*std_dev=*/1.0, (*gen)->engine());
          auto out_t = out.map_xtensor_w<EltTy>();
          *out_t = result;
        };

        switch (out.dtype()) {
          SF_UNARY_FUNCTION_CASE(float16, half_float::half);
          SF_UNARY_FUNCTION_CASE(float32, float);
          default:
            throw std::invalid_argument(
                fmt::format("Unsupported dtype({}) for operator randn",
                            out.dtype().name()));
        }
      },
      py::arg("out"), py::arg("generator") = py::none(), DOCSTRING_FILL_RANDN);

// Data-type conversion and rounding.
#define SF_DEF_CONVERT(py_name, target)                             \
  m.def(py_name, target, py::arg("input"), py::kw_only(),           \
        py::arg("dtype") = py::none(), py::arg("out") = py::none(), \
        py::arg("device_visible") = false, DOCSTRING_CONVERT)
  SF_DEF_CONVERT("convert", GenericElementwiseConvert<ConvertFunctor>);
  SF_DEF_CONVERT("ceil", GenericElementwiseConvert<ConvertCeilFunctor>);
  SF_DEF_CONVERT("floor", GenericElementwiseConvert<ConvertFloorFunctor>);
  SF_DEF_CONVERT("round", GenericElementwiseConvert<ConvertRoundFunctor>);
  SF_DEF_CONVERT("trunc", GenericElementwiseConvert<ConvertTruncFunctor>);
}

}  // namespace shortfin::python
