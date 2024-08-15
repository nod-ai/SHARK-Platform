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

namespace {

// Custom worker which hosts an asyncio event loop.
class PyWorker : public local::Worker {
 public:
  using Worker::Worker;

  void WaitForShutdown() override {
    // Need to release the GIL if blocking.
    py::gil_scoped_release g;
    Worker::WaitForShutdown();
  }

  void OnThreadStart() override {
    py::gil_scoped_acquire g;
    py::module_::import_("asyncio").attr("set_event_loop")(loop_);
  }

  void OnThreadStop() override {
    py::gil_scoped_acquire g;
    loop_.reset();
  }

  std::string to_s() { return fmt::format("PyWorker(name='{}')", name()); }

  py::object loop_;
};

PyWorker &CreatePyWorker(local::System &self, std::string name) {
  PyWorker::Options options(self.host_allocator(), std::move(name));
  auto new_worker = std::make_unique<PyWorker>(std::move(options));
  py::object worker_obj = py::cast(*new_worker.get(), py::rv_policy::reference);
  py::detail::keep_alive(worker_obj.ptr(),
                         py::cast(self, py::rv_policy::none).ptr());
  new_worker->loop_ = py::module_::import_("_shortfin.asyncio_bridge")
                          .attr("PyWorkerEventLoop")(worker_obj);

  // OnThreadStart could be called at any time after StartExistingWorker,
  // setup must be done above.
  return static_cast<PyWorker &>(
      self.StartExistingWorker(std::move(new_worker)));
}

}  // namespace

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
  // Keep weak refs to key objects that need explicit atexit shutdown.
  auto weakref = py::module_::import_("weakref");
  py::object live_system_refs = weakref.attr("WeakSet")();
  auto atexit = py::module_::import_("atexit");
  // Manually shutdown all System instances atexit if still alive (it is
  // not reliable to shutdown during interpreter finalization).
  atexit.attr("register")(py::cpp_function([](py::handle live_system_refs) {
                            for (auto it = live_system_refs.begin();
                                 it != live_system_refs.end(); ++it) {
                              (*it).attr("shutdown")();
                            }
                          }),
                          live_system_refs);

  py::class_<local::SystemBuilder>(m, "SystemBuilder")
      .def("create_system", [live_system_refs](local::SystemBuilder &self) {
        auto system_ptr = self.CreateSystem();
        auto system_obj = py::cast(system_ptr, py::rv_policy::take_ownership);
        live_system_refs.attr("add")(system_obj);
        return system_obj;
      });
  py::class_<local::System>(m, "System", py::is_weak_referenceable())
      .def("shutdown", &local::System::Shutdown)
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
      .def("create_scope", &local::System::CreateScope)
      .def("create_worker", &CreatePyWorker, py::arg("name"),
           py::rv_policy::reference_internal);

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

  py::class_<local::Worker>(m, "_Worker", py::is_weak_referenceable())
      .def("call_threadsafe", &local::Worker::CallThreadsafe)
      .def(
          "call",
          [](local::Worker &worker, py::handle callable) {
            callable.inc_ref();  // Stolen within the callback.
            auto thunk = +[](void *user_data, iree_loop_t loop,
                             iree_status_t status) noexcept -> iree_status_t {
              py::gil_scoped_acquire g;
              py::object user_callable =
                  py::steal(static_cast<PyObject *>(user_data));
              IREE_RETURN_IF_ERROR(status);
              try {
                user_callable();
              } catch (std::exception &e) {
                return iree_make_status(
                    IREE_STATUS_UNKNOWN,
                    "Python exception raised from async callback: %s",
                    e.what());
              }
              return iree_ok_status();
            };
            SHORTFIN_THROW_IF_ERROR(worker.CallLowLevel(thunk, callable.ptr()));
          })
      .def("__repr__", &local::Worker::to_s);
  py::class_<PyWorker, local::Worker>(m, "Worker")
      .def_ro("loop", &PyWorker::loop_)
      .def("__repr__", &PyWorker::to_s);
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
