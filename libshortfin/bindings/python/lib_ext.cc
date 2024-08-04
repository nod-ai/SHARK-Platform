// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./lib_ext.h"

#include "shortfin/local_system.h"
#include "shortfin/support/globals.h"
#include "shortfin/systems/host.h"

namespace shortfin::python {

NB_MODULE(lib, m) {
  m.def("initialize", shortfin::GlobalInitialize);

  BindLocalSystem(m);
  BindHostSystem(m);
}

void BindLocalSystem(py::module_ &m) {
  py::class_<LocalSystemConfig>(m, "LocalSystemConfig")
      .def("create_local_system",
           [](LocalSystemConfig &self) { return self.CreateLocalSystem(); });

  py::class_<LocalSystem>(m, "LocalSystem");
}

void BindHostSystem(py::module_ &m) {
  py::class_<systems::HostSystemConfig, LocalSystemConfig>(m,
                                                           "HostSystemConfig");
  py::class_<systems::HostCPUSystemConfig, systems::HostSystemConfig>(
      m, "HostCPUSystemConfig")
      .def(py::init<>());
}

}  // namespace shortfin::python
