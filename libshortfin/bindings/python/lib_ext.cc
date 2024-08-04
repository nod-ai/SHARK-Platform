// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./lib_ext.h"

#include "shortfin/local_system.h"
#include "shortfin/support/globals.h"
#include "shortfin/systems/amdgpu.h"
#include "shortfin/systems/host.h"

namespace shortfin::python {

NB_MODULE(lib, m) {
  m.def("initialize", shortfin::GlobalInitialize);

  BindLocalSystem(m);
  BindHostSystem(m);
  BindAMDGPUSystem(m);
}

void BindLocalSystem(py::module_ &m) {
  py::class_<LocalSystemBuilder>(m, "LocalSystemBuilder")
      .def("create_local_system",
           [](LocalSystemBuilder &self) { return self.CreateLocalSystem(); });

  py::class_<LocalSystem>(m, "LocalSystem");
}

void BindHostSystem(py::module_ &global_m) {
  auto m = global_m.def_submodule("host", "Host device management");
  py::class_<systems::HostSystemBuilder, LocalSystemBuilder>(m,
                                                             "SystemBuilder");
  py::class_<systems::HostCPUSystemBuilder, systems::HostSystemBuilder>(
      m, "CPUSystemBuilder")
      .def(py::init<>());
}

void BindAMDGPUSystem(py::module_ &global_m) {
  auto m = global_m.def_submodule("amdgpu", "AMDGPU system config");
  py::class_<systems::AMDGPUSystemBuilder, systems::HostCPUSystemBuilder>(
      m, "SystemBuilder")
      .def(py::init<>())
      .def_rw("cpu_devices_enabled",
              &systems::AMDGPUSystemBuilder::cpu_devices_enabled);
}

}  // namespace shortfin::python
