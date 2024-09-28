// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fmt/core.h>

#include "./lib_ext.h"
#include "shortfin/local/device.h"
#include "shortfin/local/fiber.h"

namespace shortfin::python {

// Casts any of int, str, local::Device, DeviceAffinity to a DeviceAffinity.
// If the object is a sequence, then the affinity is constructed from the union.
inline local::ScopedDevice CastDeviceAffinity(local::Fiber& fiber,
                                              py::handle object) {
  if (py::isinstance<local::Device>(object)) {
    return fiber.device(py::cast<local::Device*>(object));
  } else if (py::isinstance<local::DeviceAffinity>(object)) {
    return local::ScopedDevice(fiber, py::cast<local::DeviceAffinity>(object));
  } else if (py::isinstance<int>(object)) {
    return fiber.device(py::cast<int>(object));
  } else if (py::isinstance<std::string>(object)) {
    return fiber.device(py::cast<std::string>(object));
  } else if (py::isinstance<py::sequence>(object)) {
    // Important: sequence must come after string, since string is a sequence
    // and this will infinitely recurse (since the first element of the string
    // is a sequence, etc).
    local::DeviceAffinity affinity;
    for (auto item : py::cast<py::sequence>(object)) {
      affinity |= CastDeviceAffinity(fiber, item).affinity();
    }
    return local::ScopedDevice(fiber, affinity);
  }

  throw std::invalid_argument(fmt::format("Cannot cast {} to DeviceAffinity",
                                          py::repr(object).c_str()));
}

// For a bound class, binds the buffer protocol. This will result in a call
// to handler like:
//   HandlerFunctor(self, Py_buffer *view, int flags)
// This is a low level callback and must not raise any exceptions. If
// error conditions are warranted the usual PyErr_SetString approach must be
// used (and -1 returned). Return 0 on success.
template <typename CppType, typename HandlerFunctor>
void BindBufferProtocol(py::handle clazz) {
  PyBufferProcs buffer_procs;
  memset(&buffer_procs, 0, sizeof(buffer_procs));
  buffer_procs.bf_getbuffer =
      // It is not legal to raise exceptions from these callbacks.
      +[](PyObject* raw_self, Py_buffer* view, int flags) noexcept -> int {
    if (view == NULL) {
      PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
      return -1;
    }

    // Cast must succeed due to invariants.
    auto& self = py::cast<CppType&>(py::handle(raw_self));

    Py_INCREF(raw_self);
    view->obj = raw_self;
    HandlerFunctor handler;
    return handler(self, view, flags);
  };
  buffer_procs.bf_releasebuffer =
      +[](PyObject* raw_self, Py_buffer* view) noexcept -> void {};
  auto heap_type = reinterpret_cast<PyHeapTypeObject*>(clazz.ptr());
  assert(heap_type->ht_type.tp_flags & Py_TPFLAGS_HEAPTYPE &&
         "must be heap type");
  heap_type->as_buffer = buffer_procs;
}

// Represents a Py_buffer obtained via PyObject_GetBuffer() and terminated via
// PyBuffer_Release().
class PyBufferRequest {
 public:
  PyBufferRequest(py::handle& exporter, int flags) {
    int rc = PyObject_GetBuffer(exporter.ptr(), &view_, flags);
    if (rc != 0) {
      throw py::python_error();
    }
  }
  ~PyBufferRequest() { PyBuffer_Release(&view_); }
  PyBufferRequest(const PyBufferRequest&) = delete;
  void operator=(const PyBufferRequest&) = delete;

  Py_buffer& view() { return view_; }

 private:
  Py_buffer view_;
};

}  // namespace shortfin::python
