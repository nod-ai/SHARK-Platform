// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

namespace shortfin::python {
namespace py = nanobind;

void BindLocalScope(py::module_ &module);
void BindLocalSystem(py::module_ &module);
void BindHostSystem(py::module_ &module);
void BindAMDGPUSystem(py::module_ &module);

}  // namespace shortfin::python
