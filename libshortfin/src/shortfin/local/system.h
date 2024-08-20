// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_SYSTEM_H
#define SHORTFIN_LOCAL_SYSTEM_H

#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "shortfin/local/device.h"
#include "shortfin/local/worker.h"
#include "shortfin/support/api.h"
#include "shortfin/support/blocking_executor.h"
#include "shortfin/support/iree_concurrency.h"
#include "shortfin/support/iree_helpers.h"
#include "shortfin/support/stl_extras.h"

namespace shortfin::local {

namespace detail {
class BaseProcess;
}  // namespace detail

class Scope;
class System;
class SystemBuilder;

// Encapsulates resources attached to the local system. In most applications,
// there will be one of these, and it is used to keep long lived access to
// physical devices, connections, and other long lived resources which need
// to be available across the application lifetime.
//
// One does not generally construct a System by hand, instead relying
// on some form of factory that constructs one to suit both the system being
// executed on and any preferences on which resources should be accessible.
//
// As the root of the hierarchy and the owner of numerous ancillary resources,
// we declare that System is always managed via a shared_ptr, as this
// simplifies many aspects of system management.
class SHORTFIN_API System : public std::enable_shared_from_this<System> {
 public:
  System(iree_allocator_t host_allocator);
  System(const System &) = delete;
  ~System();

  // Sets a worker factory that will be used for all subsequently created
  // Worker instances. Certain bindings and integrations may need special
  // kinds of Worker classes, and this can customize that.
  void set_worker_factory(Worker::Factory factory);

  // Explicit shutdown (vs in destructor) is encouraged.
  void Shutdown();

  // Get a shared pointer from the instance.
  std::shared_ptr<System> shared_ptr() { return shared_from_this(); }

  // Access to underlying IREE API objects.
  iree_allocator_t host_allocator() { return host_allocator_; }
  iree_vm_instance_t *vm_instance() { return vm_instance_.get(); }

  // Topology access.
  std::span<const Node> nodes() { return {nodes_}; }
  std::span<Device *const> devices() { return {devices_}; }
  const std::unordered_map<std::string_view, Device *> &named_devices() {
    return named_devices_;
  }

  // Access the system wide blocking executor thread pool. This can be used
  // to execute thunks that can block on a dedicated thread and is needed
  // to bridge APIs that cannot be used in a non-blocking context.
  BlockingExecutor &blocking_executor() { return blocking_executor_; }

  // Scopes.
  // Creates a new Scope bound to this System (it will internally
  // hold a reference to this instance). All devices in system order will be
  // added to the scope.
  std::shared_ptr<Scope> CreateScope(Worker &worker,
                                     std::span<Device *const> devices);

  // Creates and starts a worker (if it is configured to run in a thread).
  Worker &CreateWorker(Worker::Options options);

  // Accesses the initialization worker that is intended to be run on the main
  // or adopted thread to perform any async interactions with the system.
  // Internally, this worker is called "__init__". It will be created on
  // demand if it does not yet exist.
  Worker &init_worker();

  // Initialization APIs. Calls to these methods is only permitted between
  // construction and Initialize().
  // ------------------------------------------------------------------------ //
  void InitializeNodes(int node_count);
  void InitializeHalDriver(std::string_view moniker,
                           iree::hal_driver_ptr driver);
  void InitializeHalDevice(std::unique_ptr<Device> device);
  void FinishInitialization();

 private:
  static std::unique_ptr<Worker> DefaultWorkerFactory(Worker::Options options);
  void AssertNotInitialized() {
    if (initialized_) {
      throw std::logic_error(
          "System::Initialize* methods can only be called during "
          "initialization");
    }
  }

  // Allocates a process in the process table and returns its new pid.
  // This is done on process construction. Note that it acquires the
  // system lock and is non-reentrant.
  int64_t AllocateProcess(detail::BaseProcess *);
  // Deallocates a process by pid. This is done on process destruction. Note
  // that is acquires the system lock and is non-reentrant.
  void DeallocateProcess(int64_t pid);

  const iree_allocator_t host_allocator_;

  string_interner interner_;
  iree::slim_mutex lock_;

  // NUMA nodes relevant to this system.
  std::vector<Node> nodes_;

  // Map of retained hal drivers. These will be released as one of the
  // last steps of destruction. There are some ancillary uses for drivers
  // after initialization, but mainly this is for keeping them alive.
  std::unordered_map<std::string_view, iree::hal_driver_ptr> hal_drivers_;

  // Map of device name to a SystemDevice.
  std::vector<std::unique_ptr<Device>> retained_devices_;
  std::unordered_map<std::string_view, Device *> named_devices_;
  std::vector<Device *> devices_;

  // VM management.
  iree::vm_instance_ptr vm_instance_;

  // Global blocking executor.
  BlockingExecutor blocking_executor_;

  // Workers.
  Worker::Factory worker_factory_ = System::DefaultWorkerFactory;
  std::vector<std::unique_ptr<Worker>> workers_;
  std::unordered_map<std::string_view, Worker *> workers_by_name_;

  // Process management.
  int next_pid_ = 1;
  std::unordered_map<int, detail::BaseProcess *> processes_by_pid_;

  // Whether initialization is complete. If true, various low level
  // mutations are disallowed.
  bool initialized_ = false;
  bool shutdown_ = false;

  friend class detail::BaseProcess;
};
using SystemPtr = std::shared_ptr<System>;

// Base class for configuration objects for setting up a System.
class SHORTFIN_API SystemBuilder {
 public:
  SystemBuilder(iree_allocator_t host_allocator)
      : host_allocator_(host_allocator) {}
  SystemBuilder() : SystemBuilder(iree_allocator_system()) {}
  virtual ~SystemBuilder() = default;

  iree_allocator_t host_allocator() { return host_allocator_; }

  // Construct a System
  virtual SystemPtr CreateSystem() = 0;

 private:
  const iree_allocator_t host_allocator_;
};

}  // namespace shortfin::local

#endif  // SHORTFIN_LOCAL_SYSTEM_H
