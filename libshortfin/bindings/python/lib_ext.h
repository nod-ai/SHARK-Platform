// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_BINDINGS_PYTHON_LIB_EXT_H
#define SHORTFIN_BINDINGS_PYTHON_LIB_EXT_H

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <span>

namespace shortfin::python {
namespace py = nanobind;

void BindArray(py::module_ &module);
void BindLocalScope(py::module_ &module);
void BindLocalSystem(py::module_ &module);
void BindHostSystem(py::module_ &module);
void BindAMDGPUSystem(py::module_ &module);

}  // namespace shortfin::python

namespace nanobind::detail {

// Type-caster for a std::span<const T>.
// Code adapted from stl/detail/nb_list.h, stl/pair.h,
// Copyright Wenzel Jakob (BSD Licensed).
//
// Note that I couldn't figure out how to get the typing to work for one
// implementation of both mutable and const spans (since the intermediate
// backing vector needs to have a mutable type). If needing a mutable span
// caster, it would be easier to just duplicate this and remove the consts.
template <typename Type>
struct type_caster<std::span<const Type>> {
  static constexpr auto Name = io_name(NB_TYPING_SEQUENCE, NB_TYPING_LIST) +
                               const_name("[") + make_caster<Type>::Name +
                               const_name("]");
  using Value = std::vector<Type>;
  using Caster = make_caster<Type>;
  template <typename T>
  using Cast = std::span<const Type>;

  Value value;
  operator std::span<const Type>() { return {value}; }

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    size_t size;
    PyObject *temp;
    /* Will initialize 'size' and 'temp'. All return values and
       return parameters are zero/NULL in the case of a failure. */
    PyObject **o = seq_get(src.ptr(), &size, &temp);
    value.clear();
    value.reserve(size);
    Caster caster;
    bool success = o != nullptr;
    flags = flags_for_local_caster<Type>(flags);
    for (size_t i = 0; i < size; ++i) {
      if (!caster.from_python(o[i], flags, cleanup) ||
          !caster.template can_cast<Type>()) {
        success = false;
        break;
      }
      value.push_back(caster.operator cast_t<Type>());
    }
    Py_XDECREF(temp);
    return success;
  }

  static handle from_cpp(std::span<const Type> src, rv_policy policy,
                         cleanup_list *cleanup) {
    object ret = steal(PyList_New(src.size()));
    if (ret.is_valid()) {
      Py_ssize_t index = 0;
      for (auto &&value : src) {
        handle h =
            Caster::from_cpp(forward_like_<Type>(value), policy, cleanup);
        if (!h.is_valid()) {
          ret.reset();
          break;
        }
        NB_LIST_SET_ITEM(ret.ptr(), index++, h.ptr());
      }
    }
    return ret.release();
  }
};

}  // namespace nanobind::detail

#endif  // SHORTFIN_BINDINGS_PYTHON_LIB_EXT_H
