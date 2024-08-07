// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./lib_ext.h"

#include "shortfin/local_scope.h"
#include "shortfin/local_system.h"
#include "shortfin/support/globals.h"
#include "shortfin/support/logging.h"
#include "shortfin/systems/amdgpu.h"
#include "shortfin/systems/host.h"

namespace shortfin::python {

NB_MODULE(lib, m) {
  m.def("initialize", shortfin::GlobalInitialize);

  BindLocalScope(m);
  BindLocalSystem(m);
  BindHostSystem(m);
  BindAMDGPUSystem(m);
}

void BindLocalSystem(py::module_ &m) {
  py::class_<LocalSystemBuilder>(m, "LocalSystemBuilder")
      .def("create_local_system",
           [](LocalSystemBuilder &self) { return self.CreateLocalSystem(); });
  py::class_<LocalSystem>(m, "LocalSystem")
      // Access devices by list, name, or lookup.
      .def_prop_ro("device_names",
                   [](LocalSystem &self) {
                     py::list names;
                     for (auto &it : self.named_devices()) {
                       names.append(it.first);
                     }
                     return names;
                   })
      .def_prop_ro("devices", &LocalSystem::devices,
                   py::rv_policy::reference_internal)
      .def(
          "device",
          [](LocalSystem &self, std::string_view key) {
            auto it = self.named_devices().find(key);
            if (it == self.named_devices().end()) {
              throw std::invalid_argument(fmt::format("No device '{}'", key));
            }
            return it->second;
          },
          py::rv_policy::reference_internal)
      .def("create_scope", &LocalSystem::CreateScope);

  // Support classes.
  py::class_<LocalNode>(m, "LocalNode")
      .def_prop_ro("node_num", &LocalNode::node_num)
      .def("__repr__", [](LocalNode &self) {
        return fmt::format("LocalNode({})", self.node_num());
      });
  py::class_<LocalDevice>(m, "LocalDevice")
      .def("name", &LocalDevice::name)
      .def("node_affinity", &LocalDevice::node_affinity)
      .def("node_locked", &LocalDevice::node_locked)
      .def("__repr__", [](py::handle self_handle) {
        auto type_name =
            py::cast<std::string>(self_handle.type().attr("__name__"));
        auto self = py::cast<LocalDevice>(self_handle);
        std::string repr = fmt::format(
            "{}(name='{}', ordinal={}:{}, node_affinity={}, node_locked={})",
            type_name, self.name(), self.address().instance_ordinal,
            self.address().queue_ordinal, self.node_affinity(),
            self.node_locked());
        return repr;
      });
}

void BindLocalScope(py::module_ &m) {
  struct DevicesSet {
    DevicesSet(LocalScope &scope) : scope(scope) {}
    LocalScope &scope;
  };
  py::class_<LocalScope>(m, "LocalScope")
      .def_prop_ro("devices", &LocalScope::devices,
                   py::rv_policy::reference_internal)
      .def_prop_ro(
          "devices", [](LocalScope &self) { return DevicesSet(self); },
          py::rv_policy::reference_internal)
      .def("device", &LocalScope::device, py::rv_policy::reference_internal)
      .def("device", &LocalScope::named_device,
           py::rv_policy::reference_internal)
      .def_prop_ro("device_names", &LocalScope::device_names)
      .def_prop_ro("named_devices", &LocalScope::named_devices,
                   py::rv_policy::reference_internal);

  py::class_<DevicesSet>(m, "_LocalScopeDevicesSet")
      .def("__len__",
           [](DevicesSet &self) { return self.scope.devices().size(); })
      .def(
          "__getitem__",
          [](DevicesSet &self, int index) { return self.scope.device(index); },
          py::rv_policy::reference_internal)
      .def(
          "__getitem__",
          [](DevicesSet &self, std::string_view name) {
            return self.scope.named_device(name);
          },
          py::rv_policy::reference_internal)
      .def(
          "__getattr__",
          [](DevicesSet &self, std::string_view name) {
            return self.scope.named_device(name);
          },
          py::rv_policy::reference_internal);
}

void BindHostSystem(py::module_ &global_m) {
  auto m = global_m.def_submodule("host", "Host device management");
  py::class_<systems::HostSystemBuilder, LocalSystemBuilder>(m,
                                                             "SystemBuilder");
  py::class_<systems::HostCPUSystemBuilder, systems::HostSystemBuilder>(
      m, "CPUSystemBuilder")
      .def(py::init<>());
  py::class_<systems::HostCPUDevice, LocalDevice>(m, "HostCPUDevice");
}

void BindAMDGPUSystem(py::module_ &global_m) {
  auto m = global_m.def_submodule("amdgpu", "AMDGPU system config");
  py::class_<systems::AMDGPUSystemBuilder, systems::HostCPUSystemBuilder>(
      m, "SystemBuilder")
      .def(py::init<>())
      .def_rw("cpu_devices_enabled",
              &systems::AMDGPUSystemBuilder::cpu_devices_enabled);
  py::class_<systems::AMDGPUDevice, LocalDevice>(m, "AMDGPUDevice");
}

}  // namespace shortfin::python
