// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

namespace shortfin::python {
namespace py = nanobind;

void BindLocalSystem(py::module_ &module);
void BindHostSystem(py::module_ &module);

}  // namespace shortfin::python
