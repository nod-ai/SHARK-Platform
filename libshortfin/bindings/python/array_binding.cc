// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./lib_ext.h"
#include "./utils.h"
#include "shortfin/array/api.h"

using namespace shortfin::array;

namespace shortfin::python {

void BindArray(py::module_ &global_m) {
  auto m = global_m.def_submodule("array", "Shortfin ND arrays");

  py::class_<DType>(m, "DType")
      .def_prop_ro("is_boolean", &DType::is_boolean)
      .def_prop_ro("is_integer", &DType::is_integer)
      .def_prop_ro("is_float", &DType::is_float)
      .def_prop_ro("is_complex", &DType::is_complex)
      .def_prop_ro("bit_count", &DType::bit_count)
      .def_prop_ro("is_byte_aligned", &DType::is_byte_aligned)
      .def_prop_ro("dense_byte_count", &DType::dense_byte_count)
      .def("is_integer_bitwidth", &DType::is_integer_bitwidth)
      .def(py::self == py::self)
      .def("__repr__", &DType::name);

  m.attr("opaque8") = DType::opaque8();
  m.attr("opaque16") = DType::opaque16();
  m.attr("opaque32") = DType::opaque32();
  m.attr("opaque64") = DType::opaque64();
  m.attr("bool8") = DType::bool8();
  m.attr("int4") = DType::int4();
  m.attr("sint4") = DType::sint4();
  m.attr("uint4") = DType::uint4();
  m.attr("int8") = DType::int8();
  m.attr("sint8") = DType::sint8();
  m.attr("uint8") = DType::uint8();
  m.attr("int16") = DType::int16();
  m.attr("sint16") = DType::sint16();
  m.attr("uint16") = DType::uint16();
  m.attr("int32") = DType::int32();
  m.attr("sint32") = DType::sint32();
  m.attr("uint32") = DType::uint32();
  m.attr("int64") = DType::int64();
  m.attr("sint64") = DType::sint64();
  m.attr("uint64") = DType::uint64();
  m.attr("float16") = DType::float16();
  m.attr("float32") = DType::float32();
  m.attr("float64") = DType::float64();
  m.attr("bfloat16") = DType::bfloat16();
  m.attr("complex64") = DType::complex64();
  m.attr("complex128") = DType::complex128();

  py::class_<storage>(m, "storage")
      .def_static(
          "allocate_host",
          [](ScopedDevice &device, iree_device_size_t allocation_size) {
            return storage::AllocateHost(device, allocation_size);
          },
          py::arg("device"), py::arg("allocation_size"), py::keep_alive<0, 1>())
      .def_static(
          "allocate_device",
          [](ScopedDevice &device, iree_device_size_t allocation_size) {
            return storage::AllocateDevice(device, allocation_size);
          },
          py::arg("device"), py::arg("allocation_size"), py::keep_alive<0, 1>())
      .def("__repr__", &storage::to_s);

  py::class_<base_array>(m, "base_array")
      .def_prop_ro("dtype", &base_array::dtype)
      .def_prop_ro("shape", &base_array::shape);
  py::class_<device_array, base_array>(m, "device_array")
      .def(py::init<storage, std::span<const size_t>, DType>(),
           py::keep_alive<0, 1>())
      .def("__init__",
           [](device_array *new_self, ScopedDevice &device,
              std::span<const size_t> shape, DType dtype) {
             new (new_self)
                 device_array(device_array::allocate(device, shape, dtype));
           })
      .def_prop_ro("device", &device_array::device,
                   py::rv_policy::reference_internal)
      .def_prop_ro("storage", &device_array::storage,
                   py::rv_policy::reference_internal)
      .def("__repr__", &device_array::to_s);
}

}  // namespace shortfin::python
