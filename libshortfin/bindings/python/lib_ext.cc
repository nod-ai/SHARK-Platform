// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./lib_ext.h"

#include "./utils.h"
#include "shortfin/local/async.h"
#include "shortfin/local/process.h"
#include "shortfin/local/program.h"
#include "shortfin/local/scope.h"
#include "shortfin/local/system.h"
#include "shortfin/local/systems/amdgpu.h"
#include "shortfin/local/systems/host.h"
#include "shortfin/support/globals.h"
#include "shortfin/support/logging.h"

namespace shortfin::python {

namespace {

class Refs {
 public:
  py::object asyncio_create_task =
      py::module_::import_("asyncio").attr("create_task");
  py::object asyncio_set_event_loop =
      py::module_::import_("asyncio").attr("set_event_loop");
  py::object asyncio_get_running_loop =
      py::module_::import_("asyncio.events").attr("get_running_loop");
  py::object asyncio_set_running_loop =
      py::module_::import_("asyncio.events").attr("_set_running_loop");
  py::object threading_Thread =
      py::module_::import_("threading").attr("Thread");
  py::object threading_current_thread =
      py::module_::import_("threading").attr("current_thread");
  py::object threading_main_thread =
      py::module_::import_("threading").attr("main_thread");

  py::handle lazy_PyWorkerEventLoop() {
    if (!lazy_PyWorkerEventLoop_.is_valid()) {
      lazy_PyWorkerEventLoop_ = py::module_::import_("_shortfin.asyncio_bridge")
                                    .attr("PyWorkerEventLoop");
    }
    return lazy_PyWorkerEventLoop_;
  }

 private:
  py::object lazy_PyWorkerEventLoop_;
};

// We need a fair bit of accounting additions in order to make Workers usable
// as asyncio loops. This extension holds that.
class PyWorkerExtension : public local::Worker::Extension {
 public:
  PyWorkerExtension(local::Worker &worker, PyInterpreterState *interp,
                    std::shared_ptr<Refs> refs)
      : Extension(worker), interp_(interp), refs_(std::move(refs)) {
    py::object worker_obj = py::cast(worker, py::rv_policy::reference);
    loop_ = refs_->lazy_PyWorkerEventLoop()(worker_obj);
  }

  static PyWorkerExtension &GetCurrent() {
    PyWorkerExtension *ext =
        local::Worker::GetCurrentExtension<PyWorkerExtension>();
    if (!ext) {
      throw std::logic_error(
          "There is no shortfin worker associated with this thread.");
    }
    return *ext;
  }

  bool initialized() { return interp_ != nullptr; }

  py::handle loop() { return loop_; }

  void OnThreadStart() noexcept override {
    // Python threading initialization.
    // If our own thread, teach Python about it. Not done for donated.
    if (worker().options().owned_thread) {
      PyThreadState_New(interp_);
    }

    py::gil_scoped_acquire g;
    // Aside from set_event_loop being old and _set_running_loop being new
    // it isn't clear to me that either can be left off.
    refs_->asyncio_set_event_loop(loop_);
    refs_->asyncio_set_running_loop(loop_);
  }

  void OnThreadStop() noexcept override {
    {
      // Do Python level thread cleanup.
      py::gil_scoped_acquire g;
      loop_.reset();

      // Scrub thread state if not donated.
      if (worker().options().owned_thread) {
        PyThreadState_Clear(PyThreadState_Get());
      } else {
        // Otherwise, juse reset the event loop.
        refs_->asyncio_set_event_loop(py::none());
        refs_->asyncio_set_running_loop(py::none());
      }
    }

    // And destroy our thread state (if not donated).
    // TODO: PyThreadState_Delete seems like it should be used here, but I
    // couldn't find that being done and I couldn't find a way to use it
    // with the GIL/thread state correct.
    if (worker().options().owned_thread) {
      PyThreadState_Swap(nullptr);
    }
  }

  // Because shutdown happens on the main thread under the purview of the
  // GIL, we must make arrangements to release it. Otherwise shutdown can
  // deadlock with processing.
  void OnBeforeShutdownWait() noexcept override {
    shutdown_wait_gil_state = PyEval_SaveThread();
  }
  void OnAfterShutdownWait() noexcept override {
    PyEval_RestoreThread(shutdown_wait_gil_state);
    shutdown_wait_gil_state = nullptr;
  }

 private:
  py::object loop_;
  PyInterpreterState *interp_ = nullptr;
  std::shared_ptr<Refs> refs_;

  PyThreadState *shutdown_wait_gil_state = nullptr;
};

class PyProcess : public local::detail::BaseProcess {
 public:
  PyProcess(std::shared_ptr<local::Scope> scope, std::shared_ptr<Refs> refs)
      : BaseProcess(std::move(scope)), refs_(std::move(refs)) {}
  using BaseProcess::Launch;

  void ScheduleOnWorker() override {
    // This is tricky: We need to retain the object reference across the
    // thread transition, but on the receiving side, the GIL will not be
    // held initially, so we must avoid any refcount maintenance until it
    // is acquired. Therefore, we manually borrow a reference and steal it in
    // the callback.
    py::handle self_object = py::cast(this, py::rv_policy::none);
    self_object.inc_ref();
    scope()->worker().CallThreadsafe(
        std::bind(&PyProcess::RunOnWorker, self_object));
  }
  static void RunOnWorker(py::handle self_handle) {
    {
      py::gil_scoped_acquire g;
      // Steal the reference back from ScheduleOnWorker. Important: this is
      // very likely the last reference to the process. So self must not be
      // touched after self_object goes out of scope.
      py::object self_object = py::steal(self_handle);
      PyProcess *self = py::cast<PyProcess *>(self_handle);
      // We assume that the run method either returns None (def) or a coroutine
      // (async def).
      auto coro = self_object.attr("run")();
      if (!coro.is_none()) {
        auto task = self->refs_->asyncio_create_task(coro);
        // Capture the self object to avoid lifetime hazzard with PyProcess
        // going away before done.
        task.attr("add_done_callback")(
            py::cpp_function([self_object](py::handle future) {
              PyProcess *done_self = py::cast<PyProcess *>(self_object);
              done_self->Terminate();
            }));
      } else {
        // Synchronous termination.
        self->Terminate();
      }
    }
  }

  std::shared_ptr<Refs> refs_;
};

py::object RunInForeground(std::shared_ptr<Refs> refs, local::System &self,
                           py::object coro) {
  bool is_main_thread =
      refs->threading_current_thread().is(refs->threading_main_thread());

  local::Worker &worker = self.init_worker();
  py::object result = py::none();
  auto done_callback = [&](py::handle future) {
    worker.Kill();
    result = future.attr("result")();
  };
  worker.CallThreadsafe([&]() {
    // Run within the worker we are about to donate to.
    py::gil_scoped_acquire g;
    auto task = refs->asyncio_create_task(coro);
    task.attr("add_done_callback")(py::cpp_function(done_callback));
  });

  auto run = py::cpp_function([&]() {
    // Release GIL and run until the worker exits.
    {
      py::gil_scoped_release g;
      worker.RunOnCurrentThread();
    }
  });

  // If running on the main thread, we spawn a background thread and join
  // it because that shields it from receiving spurious KeyboardInterrupt
  // exceptions at inopportune points.
  if (is_main_thread) {
    auto thread = refs->threading_Thread(/*group=*/py::none(), /*target=*/run);
    thread.attr("start")();
    try {
      thread.attr("join")();
    } catch (...) {
      logging::warn("Exception caught in run(). Shutting down.");
      // Leak warnings are hopeless in exceptional termination.
      py::set_leak_warnings(false);
      // Give it a go waiting for the worker thread to exit.
      worker.Kill();
      thread.attr("join")();
      self.Shutdown();
      throw;
    }
  } else {
    run();
  }

  self.Shutdown();
  return result;
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
  auto refs = std::make_shared<Refs>();
  auto worker_initializer =
      [refs, interp_state = PyInterpreterState_Get()](local::Worker &worker) {
        worker.SetExtension(
            std::make_unique<PyWorkerExtension>(worker, interp_state, refs));
      };

  py::class_<local::SystemBuilder>(m, "SystemBuilder")
      .def("create_system", [live_system_refs,
                             worker_initializer](local::SystemBuilder &self) {
        auto system_ptr = self.CreateSystem();
        system_ptr->AddWorkerInitializer(worker_initializer);
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
      .def(
          "create_scope",
          [](local::System &self, local::Worker *worker,
             py::handle raw_devices) {
            // TODO: I couldn't really figure out how to directly accept an
            // optional kw-only arg without it just being a raw object/handle.
            // If the passed devices is none, then we create the scope with
            // all devices in the system. Otherwise, with those explicitly
            // given.
            std::vector<local::Device *> devices;
            if (raw_devices.is_none()) {
              devices.assign(self.devices().begin(), self.devices().end());
            } else {
              devices = py::cast<std::vector<local::Device *>>(raw_devices);
            }

            // If no worker, default to the init worker.
            if (!worker) {
              worker = dynamic_cast<local::Worker *>(&self.init_worker());
            }

            return self.CreateScope(*worker, devices);
          },
          py::rv_policy::reference_internal,
          py::arg("worker").none() = py::none(), py::kw_only(),
          py::arg("devices") = py::none())
      .def(
          "create_worker",
          [refs](local::System &self, std::string name) -> local::Worker & {
            local::Worker::Options options(self.host_allocator(),
                                           std::move(name));
            return self.CreateWorker(options);
          },
          py::arg("name"), py::rv_policy::reference_internal)
      .def(
          "run",
          [refs](local::System &self, py::object coro) {
            return RunInForeground(refs, self, std::move(coro));
          },
          py::arg("coro"))
      // Methods not on System but on child objects, taking System as an arg.
      // Emitted here for convenience.
      .def("load_module", &local::ProgramModule::Load, py::arg("path"),
           py::arg("mmap") = true);

  // Support classes.
  py::class_<local::Node>(m, "Node")
      .def_prop_ro("node_num", &local::Node::node_num)
      .def("__repr__", [](local::Node &self) {
        return fmt::format("local::Node({})", self.node_num());
      });
  py::class_<local::Device>(m, "Device")
      .def_prop_ro("name", &local::Device::name)
      .def_prop_ro("node_affinity", &local::Device::node_affinity)
      .def_prop_ro("node_locked", &local::Device::node_locked)
      .def(py::self == py::self)
      .def("__repr__", &local::Device::to_s);
  py::class_<local::DeviceAffinity>(m, "DeviceAffinity")
      .def(py::init<>())
      .def(py::init<local::Device *>())
      .def(py::self == py::self)
      .def("add", &local::DeviceAffinity::AddDevice)
      .def("__add__", &local::DeviceAffinity::AddDevice)
      .def("__repr__", &local::DeviceAffinity::to_s);

  py::class_<local::Program>(m, "Program");
  py::class_<local::ProgramModule>(m, "ProgramModule")
      .def("__repr__", &local::ProgramModule::to_s)
      .def_static("load", &local::ProgramModule::Load, py::arg("system"),
                  py::arg("path"), py::arg("mmap") = true);

  struct DevicesSet {
    DevicesSet(local::Scope &scope) : scope(scope) {}
    local::Scope &scope;
  };
  py::class_<local::Scope>(m, "Scope")
      .def("__repr__", &local::Scope::to_s)
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
          py::rv_policy::reference_internal)
      .def(
          "load_unbound_program",
          [](local::Scope &scope, std::span<const local::ProgramModule> modules,
             bool trace_execution) {
            local::Program::Options options;
            options.trace_execution = trace_execution;
            return scope.LoadUnboundProgram(modules, std::move(options));
          },
          py::arg("modules"), py::arg("trace_execution") = false);

  py::class_<local::ScopedDevice>(m, "ScopedDevice")
      .def_prop_ro("scope", &local::ScopedDevice::scope,
                   py::rv_policy::reference)
      .def_prop_ro("affinity", &local::ScopedDevice::affinity,
                   py::rv_policy::reference_internal)
      .def_prop_ro("raw_device", &local::ScopedDevice::raw_device,
                   py::rv_policy::reference_internal)
      .def(py::self == py::self)
      .def("__await__",
           [](local::ScopedDevice &self) {
             py::object future = py::cast(self.OnSync(), py::rv_policy::move);
             return future.attr("__await__")();
           })
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

  py::class_<local::Worker>(m, "Worker", py::is_weak_referenceable())
      .def_prop_ro("loop",
                   [](local::Worker &self) {
                     auto *ext = self.GetExtension<PyWorkerExtension>();
                     if (!ext) {
                       throw std::logic_error(
                           "Worker was not initialized for access from Python");
                     }
                     return ext->loop();
                   })
      .def("call_threadsafe", &local::Worker::CallThreadsafe)
      .def("call",
           [](local::Worker &self, py::handle callable) {
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
             SHORTFIN_THROW_IF_ERROR(self.CallLowLevel(thunk, callable.ptr()));
           })
      .def("delay_call",
           [](local::Worker &self, iree_time_t deadline_ns,
              py::handle callable) {
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
             SHORTFIN_THROW_IF_ERROR(self.WaitUntilLowLevel(
                 iree_make_deadline(deadline_ns), thunk, callable.ptr()));
           })
      .def("_delay_to_deadline_ns",
           [](local::Worker &self, double delay_seconds) {
             return self.ConvertRelativeTimeoutToDeadlineNs(
                 static_cast<iree_duration_t>(delay_seconds * 1e9));
           })
      .def("_now", [](local::Worker &self) { return self.now(); })
      .def("__repr__", &local::Worker::to_s);

  py::class_<PyProcess>(m, "Process")
      .def("__init__", [](py::args, py::kwargs) {})
      .def_static(
          "__new__",
          [refs](py::handle py_type, py::args,
                 std::shared_ptr<local::Scope> scope, py::kwargs) {
            return custom_new<PyProcess>(py_type, std::move(scope), refs);
          },
          py::arg("type"), py::arg("args"), py::arg("scope"), py::arg("kwargs"))
      .def_prop_ro("pid", &PyProcess::pid)
      .def_prop_ro("scope", &PyProcess::scope)
      .def("launch",
           [](py::object self_obj) {
             PyProcess &self = py::cast<PyProcess &>(self_obj);
             self.Launch();
             return self_obj;
           })
      .def("__await__",
           [](PyProcess &self) {
             py::object future =
                 py::cast(local::CompletionEvent(self.OnTermination()),
                          py::rv_policy::move);
             return future.attr("__await__")();
           })
      .def("__repr__", &PyProcess::to_s);

  py::class_<local::CompletionEvent>(m, "CompletionEvent")
      .def(py::init<>())
      .def("__await__", [](py::handle self_obj) {
        auto &worker_ext = PyWorkerExtension::GetCurrent();
        auto &self = py::cast<local::CompletionEvent &>(self_obj);
        py::object future = worker_ext.loop().attr("create_future")();
        // Stashing self as an attribute on the future, keeps us alive,
        // which transitively means that the wait source we get from self
        // stays alive. This can be done better later with a custom
        // Future.
        future.attr("_sf_event") = self_obj;
        py::object iter_ret = future.attr("__iter__")();

        // Pass the future object as void* user data to the C callback
        // interface. This works because we release the object pointer going
        // in and then steal it back to a py::object once the GIL has been
        // acquired (since dec_ref'ing it must happen within the GIL).
        // Because the self object was stashed on an attribute above, the
        // wait_source is valid for the entire sequence.
        SHORTFIN_THROW_IF_ERROR(worker_ext.worker().WaitOneLowLevel(
            /*wait_source=*/
            self, iree_infinite_timeout(),
            +[](void *future_vp, iree_loop_t loop,
                iree_status_t status) noexcept -> iree_status_t {
              py::gil_scoped_acquire g;
              py::object future = py::steal(static_cast<PyObject *>(future_vp));
              try {
                SHORTFIN_THROW_IF_ERROR(status);
                future.attr("set_result")(py::none());
              } catch (std::exception &e) {
                auto RuntimeError = py::handle(PyExc_RuntimeError);
                future.attr("set_exception")(RuntimeError(e.what()));
              }
              return iree_ok_status();
            },
            static_cast<void *>(future.release().ptr())));
        return iter_ret;
      });
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
