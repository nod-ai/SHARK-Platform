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

namespace {
static const char DOCSTRING_ARRAY_COPY_FROM[] =
    R"(Copy contents from a source array to this array.

Equivalent to `dest_array.storage.copy_from(source_array.storage)`.
)";

static const char DOCSTRING_ARRAY_COPY_TO[] =
    R"(Copy contents this array to a destination array.

Equivalent to `dest_array.storage.copy_from(source_array.storage)`.
)";

static const char DOCSTRING_ARRAY_FILL[] = R"(Fill an array with a value.

Equivalent to `array.storage.fill(pattern)`.
)";

static const char DOCSTRING_STORAGE_DATA[] = R"(Access raw binary contents.

Accessing `foo = storage.data` is equivalent to `storage.data.map(read=True)`.
The returned object is a context manager that will close on exit.

Assigning `storage.data = array.array("f", [1.0])` will copy that raw data
from the source object using the buffer protocol. The source data must be
less than or equal to the length of the storage object. Note that the entire
storage is mapped as write-only/discardable, and writing less than the storage
bytes leaves any unwritten contents in an undefined state.

As with `map`, this will only work on buffers that are host visible, which
includes all host buffers and device buffers created with the necessary access.
)";

static const char DOCSTRING_STORAGE_COPY_FROM[] =
    R"(Copy contents from a source storage to this array.

This operation executes asynchronously and the effect will only be visible
once the execution scope has been synced to the point of mutation.
)";

static const char DOCSTRING_STORAGE_FILL[] = R"(Fill a storage with a value.

Takes as argument any value that can be interpreted as a buffer with the Python
buffer protocol of size 1, 2, or 4 bytes. The storage will be filled uniformly
with the pattern.

This operation executes asynchronously and the effect will only be visible
once the execution scope has been synced to the point of mutation.
)";

static const char DOCSTRING_STORAGE_MAP[] =
    R"(Create a mapping of the buffer contents in host memory.

Support kwargs of:

read: Enables read access to the mapped memory.
write: Enables write access to the mapped memory and will flush upon close
  (for non-unified memory systems).
discard: Indicates that the entire memory map should be treated as if it will
  be overwritten. Initial contents will be undefined.

Mapping memory for access from the host requires a compatible buffer that has
been created with host visibility (which includes host buffers).

The returned mapping object is a context manager that will close/flush on
exit. Alternatively, the `close()` method can be invoked explicitly.
)";

// Does in-place creation of a mapping object and stores a pointer to the
// contained array::mapping C++ object.
py::object CreateMappingObject(mapping **out_cpp_mapping) {
  py::object py_mapping = py::inst_alloc(py::type<mapping>());
  mapping *cpp_mapping = py::inst_ptr<mapping>(py_mapping);
  new (cpp_mapping) mapping();
  py::inst_mark_ready(py_mapping);
  *out_cpp_mapping = cpp_mapping;
  return py_mapping;
}

}  // namespace

void BindArray(py::module_ &m) {
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

  // storage
  py::class_<storage>(m, "storage")
      .def_static(
          "allocate_host",
          [](local::ScopedDevice &device, iree_device_size_t allocation_size) {
            return storage::allocate_host(device, allocation_size);
          },
          py::arg("device"), py::arg("allocation_size"), py::keep_alive<0, 1>())
      .def_static(
          "allocate_device",
          [](local::ScopedDevice &device, iree_device_size_t allocation_size) {
            return storage::allocate_device(device, allocation_size);
          },
          py::arg("device"), py::arg("allocation_size"), py::keep_alive<0, 1>())
      .def(
          "fill",
          [](storage &self, py::handle buffer) {
            Py_buffer py_view;
            int flags = PyBUF_FORMAT | PyBUF_ND;  // C-Contiguous ND.
            if (PyObject_GetBuffer(buffer.ptr(), &py_view, flags) != 0) {
              throw py::python_error();
            }
            PyBufferReleaser py_view_releaser(py_view);
            self.fill(py_view.buf, py_view.len);
          },
          py::arg("pattern"), DOCSTRING_STORAGE_FILL)
      .def(
          "copy_from", [](storage &self, storage &src) { self.copy_from(src); },
          py::arg("source_storage"), DOCSTRING_STORAGE_COPY_FROM)
      .def(
          "map",
          [](storage &self, bool read, bool write, bool discard) {
            int access = 0;
            if (read) access |= IREE_HAL_MEMORY_ACCESS_READ;
            if (write) access |= IREE_HAL_MEMORY_ACCESS_WRITE;
            if (discard) access |= IREE_HAL_MEMORY_ACCESS_DISCARD;
            if (!access) {
              throw std::invalid_argument(
                  "One of the access flags must be set");
            }
            mapping *cpp_mapping = nullptr;
            py::object py_mapping = CreateMappingObject(&cpp_mapping);
            self.map_explicit(
                *cpp_mapping,
                static_cast<iree_hal_memory_access_bits_t>(access));
            return py_mapping;
          },
          py::kw_only(), py::arg("read") = false, py::arg("write") = false,
          py::arg("discard") = false, DOCSTRING_STORAGE_MAP)
      // The 'data' prop is a short-hand for accessing the backing storage
      // in a one-shot manner (as for reading or writing). Getting the attribute
      // will map for read and return a memory view (equiv to map(read=True)).
      // On write, it will accept an object implementing the buffer protocol
      // and write/discard the backing storage.
      .def_prop_rw(
          "data",
          [](storage &self) {
            mapping *cpp_mapping = nullptr;
            py::object py_mapping = CreateMappingObject(&cpp_mapping);
            *cpp_mapping = self.map_read();
            return py_mapping;
          },
          [](storage &self, py::handle buffer_obj) {
            PyBufferRequest src_info(buffer_obj, PyBUF_SIMPLE);
            auto dest_data = self.map_write_discard();
            if (src_info.view().len > dest_data.size()) {
              throw std::invalid_argument(
                  fmt::format("Cannot write {} bytes into buffer of {} bytes",
                              src_info.view().len, dest_data.size()));
            }
            std::memcpy(dest_data.data(), src_info.view().buf,
                        src_info.view().len);
          },
          DOCSTRING_STORAGE_DATA)
      .def("__repr__", &storage::to_s);

  // mapping
  auto mapping_class = py::class_<mapping>(m, "mapping");
  mapping_class.def("close", &mapping::reset)
      .def_prop_ro("valid", [](mapping &self) -> bool { return self; })
      .def("__enter__", [](py::object self_obj) { return self_obj; })
      .def(
          "__exit__",
          [](mapping &self, py::handle exc_type, py::handle exc_value,
             py::handle exc_tb) { self.reset(); },
          py::arg("exc_type").none(), py::arg("exc_value").none(),
          py::arg("exc_tb").none());
  struct MappingBufferHandler {
    int operator()(mapping &self, Py_buffer *view, int flags) {
      view->buf = self.data();
      view->len = self.size();
      view->readonly = self.writable();
      view->itemsize = 1;
      view->format = (char *)"B";  // Byte
      view->ndim = 1;
      view->shape = nullptr;
      view->strides = nullptr;
      view->suboffsets = nullptr;
      view->internal = nullptr;
      return 0;
    }
  };
  BindBufferProtocol<mapping, MappingBufferHandler>(mapping_class);

  // base_array and subclasses
  py::class_<base_array>(m, "base_array")
      .def_prop_ro("dtype", &base_array::dtype)
      .def_prop_ro("shape", &base_array::shape);
  py::class_<device_array, base_array>(m, "device_array")
      .def("__init__", [](py::args, py::kwargs) {})
      .def_static("__new__",
                  [](py::handle py_type, class storage storage,
                     std::span<const size_t> shape, DType dtype) {
                    return custom_new_keep_alive<device_array>(
                        py_type, /*keep_alive=*/storage.scope(), storage, shape,
                        dtype);
                  })
      .def_static("__new__",
                  [](py::handle py_type, local::ScopedDevice &device,
                     std::span<const size_t> shape, DType dtype) {
                    return custom_new_keep_alive<device_array>(
                        py_type, /*keep_alive=*/device.scope(),
                        device_array::for_device(device, shape, dtype));
                  })
      .def_static("for_device",
                  [](local::ScopedDevice &device, std::span<const size_t> shape,
                     DType dtype) {
                    return custom_new_keep_alive<device_array>(
                        py::type<device_array>(), /*keep_alive=*/device.scope(),
                        device_array::for_device(device, shape, dtype));
                  })
      .def_static("for_host",
                  [](local::ScopedDevice &device, std::span<const size_t> shape,
                     DType dtype) {
                    return custom_new_keep_alive<device_array>(
                        py::type<device_array>(), /*keep_alive=*/device.scope(),
                        device_array::for_host(device, shape, dtype));
                  })
      .def("for_transfer",
           [](device_array &self) {
             return custom_new_keep_alive<device_array>(
                 py::type<device_array>(),
                 /*keep_alive=*/self.device().scope(), self.for_transfer());
           })
      .def_prop_ro("device", &device_array::device,
                   py::rv_policy::reference_internal)
      .def_prop_ro("storage", &device_array::storage,
                   py::rv_policy::reference_internal)

      .def(
          "fill",
          [](py::handle_t<device_array> self, py::handle buffer) {
            self.attr("storage").attr("fill")(buffer);
          },
          py::arg("pattern"), DOCSTRING_ARRAY_FILL)
      .def("copy_from", &device_array::copy_from, py::arg("source_array"),
           DOCSTRING_ARRAY_COPY_FROM)
      .def("copy_to", &device_array::copy_to, py::arg("dest_array"),
           DOCSTRING_ARRAY_COPY_TO)
      .def("__repr__", &device_array::to_s);
}

}  // namespace shortfin::python
