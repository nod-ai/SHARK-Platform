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

static const char DOCSTRING_TRANSPOSE[] =
    R"(Transposes axes of an array according to a permutation vector.

Args:
  input: Array to transpose.
  permutation: New sequence of axes. Must have same number of elements as the
    rank of input.
  out: If given, then the results are written to this array.
  device_visible: Whether to make the result array visible to devices. Defaults
    to False.
)";

#define SF_UNARY_FUNCTION_CASE(dtype_name, cpp_type) \
  case DType::dtype_name():                          \
    return compute.template operator()<cpp_type>()

#define SF_UNARY_THUNK_CASE(dtype_name, cpp_type) \
  case DType::dtype_name():                       \
    compute.template operator()<cpp_type>();      \
    break

#define SF_MOVEMENT_OP_SWITCH(dtype)                                     \
  if (!dtype.is_byte_aligned())                                          \
    throw std::invalid_argument(                                         \
        "data movement ops are only defined for byte aligned dtypes");   \
  switch (dtype.dense_byte_count()) {                                    \
    case 1:                                                              \
      return compute.template operator()<uint8_t>();                     \
    case 2:                                                              \
      return compute.template operator()<uint16_t>();                    \
    case 4:                                                              \
      return compute.template operator()<uint32_t>();                    \
    case 8:                                                              \
      return compute.template operator()<uint64_t>();                    \
    default:                                                             \
      throw std::invalid_argument(                                       \
          "data movement ops are only defined for dtypes of size 1, 2, " \
          "4, 8");                                                       \
  }

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

void OptionalArrayCast(py::handle handle,
                       std::optional<device_array> &maybe_array) {
  if (py::isinstance<device_array>(handle)) {
    maybe_array.emplace(py::cast<device_array>(handle));
  }
}

int DTypePromotionRank(DType dtype) {
  int rank = 1;
  if (dtype.is_boolean())
    rank *= 1000;
  else if (dtype.is_integer())
    rank *= 2000;
  else if (dtype.is_float())
    rank *= 4000;
  else if (dtype.is_complex())
    rank *= 8000;
  return rank + dtype.bit_count();
}

DType PromoteArithmeticTypes(std::optional<DType> lhs_dtype,
                             std::optional<DType> rhs_dtype) {
  if (!lhs_dtype && !rhs_dtype) {
    throw std::invalid_argument(
        "Elementwsie operators require at least one argument to be a "
        "device_array");
  }

  // One not an array: promote to the array type.
  if (!lhs_dtype)
    return *rhs_dtype;
  else if (!rhs_dtype)
    return *lhs_dtype;

  int lhs_rank = DTypePromotionRank(*lhs_dtype);
  int rhs_rank = DTypePromotionRank(*rhs_dtype);
  if (lhs_rank < rhs_rank)
    return *rhs_dtype;
  else
    return *lhs_dtype;
}

// ---------------------------------------------------------------------------//
// Elementwise support
// ---------------------------------------------------------------------------//

// Python element type scalar conversion functions.
uint8_t ConvertPyToEltTy(py::handle py_value, uint8_t zero) {
  return py::cast<uint8_t>(py_value);
}

int8_t ConvertPyToEltTy(py::handle py_value, int8_t zero) {
  return py::cast<int8_t>(py_value);
}

uint16_t ConvertPyToEltTy(py::handle py_value, uint16_t zero) {
  return py::cast<uint16_t>(py_value);
}

int16_t ConvertPyToEltTy(py::handle py_value, int16_t zero) {
  return py::cast<int16_t>(py_value);
}

uint32_t ConvertPyToEltTy(py::handle py_value, uint32_t zero) {
  return py::cast<uint32_t>(py_value);
}

int32_t ConvertPyToEltTy(py::handle py_value, int32_t zero) {
  return py::cast<int32_t>(py_value);
}

uint64_t ConvertPyToEltTy(py::handle py_value, uint64_t zero) {
  return py::cast<uint64_t>(py_value);
}

int64_t ConvertPyToEltTy(py::handle py_value, int64_t zero) {
  return py::cast<int64_t>(py_value);
}

float ConvertPyToEltTy(py::handle py_value, float zero) {
  return py::cast<float>(py_value);
}

double ConvertPyToEltTy(py::handle py_value, double zero) {
  return py::cast<double>(py_value);
}

half_float::half ConvertPyToEltTy(py::handle py_value, half_float::half zero) {
  // Python can't cast directly to half so first go to double.
  return static_cast<half_float::half>(py::cast<double>(py_value));
}

struct AddFunctor {
  template <typename Lhs, typename Rhs>
  static auto Invoke(Lhs &&lhs, Rhs &&rhs) {
    return lhs + rhs;
  }
};

struct DivideFunctor {
  template <typename Lhs, typename Rhs>
  static auto Invoke(Lhs &&lhs, Rhs &&rhs) {
    return lhs / rhs;
  }
};

struct MultiplyFunctor {
  template <typename Lhs, typename Rhs>
  static auto Invoke(Lhs &&lhs, Rhs &&rhs) {
    return lhs * rhs;
  }
};

struct SubtractFunctor {
  template <typename Lhs, typename Rhs>
  static auto Invoke(Lhs &&lhs, Rhs &&rhs) {
    return lhs - rhs;
  }
};

template <typename ElementwiseFunctor>
device_array ElementwiseOperation(py::handle lhs, py::handle rhs,
                                  std::optional<device_array> out,
                                  bool device_visible) {
  std::optional<device_array> lhs_array;
  OptionalArrayCast(lhs, lhs_array);
  std::optional<device_array> rhs_array;
  OptionalArrayCast(rhs, rhs_array);
  auto dtype = PromoteArithmeticTypes(
      lhs_array ? std::optional<DType>(lhs_array->dtype()) : std::nullopt,
      rhs_array ? std::optional<DType>(rhs_array->dtype()) : std::nullopt);
  if (lhs_array && lhs_array->dtype() != dtype) {
    auto converted = GenericElementwiseConvert<ConvertFunctor>(
        *lhs_array, dtype, /*out=*/std::nullopt,
        /*device_visible=*/false);
    lhs_array.reset();
    lhs_array.emplace(std::move(converted));
  }
  if (rhs_array && rhs_array->dtype() != dtype) {
    auto converted = GenericElementwiseConvert<ConvertFunctor>(
        *rhs_array, dtype, /*out=*/std::nullopt,
        /*device_visible=*/false);
    rhs_array.reset();
    rhs_array.emplace(std::move(converted));
  }

  auto compute = [&]<typename EltTy>() -> device_array {
    auto handle_result = [&]<typename D, typename A>(
                             D &&device, A &&result) -> device_array {
      if (!out) {
        out.emplace(device_array::for_host(device, result.shape(), dtype,
                                           device_visible));
      }
      auto out_t = out->map_xtensor_w<EltTy>();
      *out_t = result;
      return *out;
    };
    if (!rhs_array) {
      auto lhs_t = lhs_array->map_xtensor<EltTy>();
      xt::xarray<EltTy> rhs_scalar = ConvertPyToEltTy(rhs, EltTy());
      return handle_result(lhs_array->device(),
                           ElementwiseFunctor::Invoke(*lhs_t, rhs_scalar));
    } else if (!lhs_array) {
      xt::xarray<EltTy> lhs_scalar = ConvertPyToEltTy(lhs, EltTy());
      auto rhs_t = rhs_array->map_xtensor<EltTy>();
      return handle_result(rhs_array->device(),
                           ElementwiseFunctor::Invoke(lhs_scalar, *rhs_t));
    } else {
      auto lhs_t = lhs_array->map_xtensor<EltTy>();
      auto rhs_t = rhs_array->map_xtensor<EltTy>();
      return handle_result(lhs_array->device(),
                           ElementwiseFunctor::Invoke(*lhs_t, *rhs_t));
    }
  };

  switch (dtype) {
    SF_UNARY_FUNCTION_CASE(float16, half_float::half);
    SF_UNARY_FUNCTION_CASE(float32, float);
    SF_UNARY_FUNCTION_CASE(float64, double);
    SF_UNARY_FUNCTION_CASE(uint8, uint8_t);
    SF_UNARY_FUNCTION_CASE(int8, int8_t);
    SF_UNARY_FUNCTION_CASE(uint16, uint16_t);
    SF_UNARY_FUNCTION_CASE(int16, int16_t);
    SF_UNARY_FUNCTION_CASE(uint32, uint32_t);
    SF_UNARY_FUNCTION_CASE(int32, uint32_t);
    SF_UNARY_FUNCTION_CASE(uint64, uint64_t);
    SF_UNARY_FUNCTION_CASE(int64, int64_t);
    default:
      throw std::invalid_argument(fmt::format(
          "Unsupported dtype({}) for in elementwise op", dtype.name()));
  }
}

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

  // Transpose.
  m.def(
      "transpose",
      [](device_array input, std::vector<size_t> permutation,
         std::optional<device_array> out, bool device_visible) {
        auto compute = [&]<typename EltTy>() -> device_array {
          auto input_t = input.map_xtensor<EltTy>();
          auto permuted_t =
              xt::transpose(*input_t, permutation, xt::check_policy::full());
          if (!out) {
            out.emplace(device_array::for_host(input.device(),
                                               permuted_t.shape(),
                                               input.dtype(), device_visible));
          }
          auto out_t = out->map_xtensor_w<EltTy>();
          *out_t = permuted_t;
          return *out;
        };
        SF_MOVEMENT_OP_SWITCH(input.dtype());
      },
      py::arg("input"), py::arg("permutation"), py::arg("out") = py::none(),
      py::arg("device_visible") = false, DOCSTRING_TRANSPOSE);

// Elementwise.
#define SF_DEF_ELEMENTWISE(py_name, target)                             \
  m.def(py_name, target, py::arg("lhs"), py::arg("rhs"), py::kw_only(), \
        py::arg("out") = py::none(), py::arg("device_visible") = false)
  SF_DEF_ELEMENTWISE("add", ElementwiseOperation<AddFunctor>);
  SF_DEF_ELEMENTWISE("divide", ElementwiseOperation<DivideFunctor>);
  SF_DEF_ELEMENTWISE("multiply", ElementwiseOperation<MultiplyFunctor>);
  SF_DEF_ELEMENTWISE("subtract", ElementwiseOperation<SubtractFunctor>);

}  // namespace shortfin::python

}  // namespace shortfin::python
