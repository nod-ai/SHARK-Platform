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
  py::class_<LocalSystemConfig>(m, "LocalSystemConfig")
      .def("create_local_system",
           [](LocalSystemConfig &self) { return self.CreateLocalSystem(); });

  py::class_<LocalSystem>(m, "LocalSystem");
}

void BindHostSystem(py::module_ &global_m) {
  auto m = global_m.def_submodule("host", "Host device management");
  py::class_<systems::HostSystemConfig, LocalSystemConfig>(m, "SystemConfig");
  py::class_<systems::HostCPUSystemConfig, systems::HostSystemConfig>(
      m, "CPUSystemConfig")
      .def(py::init<>());
}

void BindAMDGPUSystem(py::module_ &global_m) {
  auto m = global_m.def_submodule("amdgpu", "AMDGPU system config");
  py::class_<systems::AMDGPUSystemConfig, systems::HostCPUSystemConfig>(
      m, "SystemConfig")
      .def(py::init<>())
      .def_rw("cpu_devices_enabled",
              &systems::AMDGPUSystemConfig::cpu_devices_enabled);
}

}  // namespace shortfin::python
