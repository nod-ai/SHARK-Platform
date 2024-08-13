// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fmt/core.h>

#include "./lib_ext.h"
#include "shortfin/local/device.h"
#include "shortfin/local/scope.h"

namespace shortfin::python {

// Casts any of int, str, LocalDevice, DeviceAffinity to a DeviceAffinity.
// If the object is a sequence, then the affinity is constructed from the union.
inline ScopedDevice CastDeviceAffinity(LocalScope &scope, py::handle object) {
  if (py::isinstance<LocalDevice>(object)) {
    return scope.device(py::cast<LocalDevice *>(object));
  } else if (py::isinstance<DeviceAffinity>(object)) {
    return ScopedDevice(scope, py::cast<DeviceAffinity>(object));
  } else if (py::isinstance<int>(object)) {
    return scope.device(py::cast<int>(object));
  } else if (py::isinstance<std::string>(object)) {
    return scope.device(py::cast<std::string>(object));
  } else if (py::isinstance<py::sequence>(object)) {
    // Important: sequence must come after string, since string is a sequence
    // and this will infinitely recurse (since the first element of the string
    // is a sequence, etc).
    DeviceAffinity affinity;
    for (auto item : py::cast<py::sequence>(object)) {
      affinity |= CastDeviceAffinity(scope, item).affinity();
    }
    return ScopedDevice(scope, affinity);
  }

  throw std::invalid_argument(fmt::format("Cannot cast {} to DeviceAffinity",
                                          py::repr(object).c_str()));
}

}  // namespace shortfin::python
