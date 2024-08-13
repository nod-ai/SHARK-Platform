// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./lib_ext.h"

#include "./utils.h"
#include "shortfin/local/scope.h"
#include "shortfin/local/system.h"
#include "shortfin/local/systems/amdgpu.h"
#include "shortfin/local/systems/host.h"
#include "shortfin/support/globals.h"
#include "shortfin/support/logging.h"

namespace shortfin::python {

NB_MODULE(lib, m) {
  m.def("initialize", shortfin::GlobalInitialize);
  auto local_m = m.def_submodule("local");
  BindLocal(local_m);
  BindHostSystem(local_m);
  BindAMDGPUSystem(local_m);

  auto array_m = m.def_submodule("array");
  BindArray(array_m);
}

void BindLocal(py::module_ &m) {
  py::class_<local::SystemBuilder>(m, "SystemBuilder")
      .def("create_system",
           [](local::SystemBuilder &self) { return self.CreateSystem(); });
  py::class_<local::System>(m, "System")
      // Access devices by list, name, or lookup.
      .def_prop_ro("device_names",
                   [](local::System &self) {
                     py::list names;
                     for (auto &it : self.named_devices()) {
                       names.append(it.first);
                     }
                     return names;
                   })
      .def_prop_ro("devices", &local::System::devices,
                   py::rv_policy::reference_internal)
      .def(
          "device",
          [](local::System &self, std::string_view key) {
            auto it = self.named_devices().find(key);
            if (it == self.named_devices().end()) {
              throw std::invalid_argument(fmt::format("No device '{}'", key));
            }
            return it->second;
          },
          py::rv_policy::reference_internal)
      .def("create_scope", &local::System::CreateScope);

  // Support classes.
  py::class_<local::Node>(m, "Node")
      .def_prop_ro("node_num", &local::Node::node_num)
      .def("__repr__", [](local::Node &self) {
        return fmt::format("local::Node({})", self.node_num());
      });
  py::class_<local::Device>(m, "Device")
      .def("name", &local::Device::name)
      .def("node_affinity", &local::Device::node_affinity)
      .def("node_locked", &local::Device::node_locked)
      .def(py::self == py::self)
      .def("__repr__", &local::Device::to_s);
  py::class_<local::DeviceAffinity>(m, "DeviceAffinity")
      .def(py::init<>())
      .def(py::init<local::Device *>())
      .def(py::self == py::self)
      .def("add", &local::DeviceAffinity::AddDevice)
      .def("__add__", &local::DeviceAffinity::AddDevice)
      .def("__repr__", &local::DeviceAffinity::to_s);

  struct DevicesSet {
    DevicesSet(local::Scope &scope) : scope(scope) {}
    local::Scope &scope;
  };
  py::class_<local::Scope>(m, "Scope")
      .def_prop_ro("raw_devices", &local::Scope::raw_devices,
                   py::rv_policy::reference_internal)
      .def(
          "raw_device",
          [](local::Scope &self, int index) { return self.raw_device(index); },
          py::rv_policy::reference_internal)
      .def(
          "raw_device",
          [](local::Scope &self, std::string_view name) {
            return self.raw_device(name);
          },
          py::rv_policy::reference_internal)
      .def_prop_ro(
          "devices", [](local::Scope &self) { return DevicesSet(self); },
          py::rv_policy::reference_internal)
      .def_prop_ro("device_names", &local::Scope::device_names)
      .def_prop_ro("named_devices", &local::Scope::named_devices,
                   py::rv_policy::reference_internal)
      .def(
          "device",
          [](local::Scope &self, py::args args) {
            return CastDeviceAffinity(self, args);
          },
          py::rv_policy::reference_internal);
  py::class_<local::ScopedDevice>(m, "ScopedDevice")
      .def_prop_ro("scope", &local::ScopedDevice::scope,
                   py::rv_policy::reference)
      .def_prop_ro("affinity", &local::ScopedDevice::affinity,
                   py::rv_policy::reference_internal)
      .def_prop_ro("raw_device", &local::ScopedDevice::raw_device,
                   py::rv_policy::reference_internal)
      .def(py::self == py::self)
      .def("__repr__", &local::ScopedDevice::to_s);

  py::class_<DevicesSet>(m, "_ScopeDevicesSet")
      .def("__len__",
           [](DevicesSet &self) { return self.scope.raw_devices().size(); })
      .def(
          "__getitem__",
          [](DevicesSet &self, int index) { return self.scope.device(index); },
          py::rv_policy::reference_internal)
      .def(
          "__getitem__",
          [](DevicesSet &self, std::string_view name) {
            return self.scope.device(name);
          },
          py::rv_policy::reference_internal)
      .def(
          "__getattr__",
          [](DevicesSet &self, std::string_view name) {
            return self.scope.device(name);
          },
          py::rv_policy::reference_internal);
}

void BindHostSystem(py::module_ &global_m) {
  auto m = global_m.def_submodule("host", "Host device management");
  py::class_<local::systems::HostSystemBuilder, local::SystemBuilder>(
      m, "SystemBuilder");
  py::class_<local::systems::HostCPUSystemBuilder,
             local::systems::HostSystemBuilder>(m, "CPUSystemBuilder")
      .def(py::init<>());
  py::class_<local::systems::HostCPUDevice, local::Device>(m, "HostCPUDevice");
}

void BindAMDGPUSystem(py::module_ &global_m) {
  auto m = global_m.def_submodule("amdgpu", "AMDGPU system config");
  py::class_<local::systems::AMDGPUSystemBuilder,
             local::systems::HostCPUSystemBuilder>(m, "SystemBuilder")
      .def(py::init<>())
      .def_rw("cpu_devices_enabled",
              &local::systems::AMDGPUSystemBuilder::cpu_devices_enabled);
  py::class_<local::systems::AMDGPUDevice, local::Device>(m, "AMDGPUDevice");
}

}  // namespace shortfin::python
