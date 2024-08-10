// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_SCOPE_H
#define SHORTFIN_LOCAL_SCOPE_H

#include <functional>
#include <span>
#include <unordered_map>

#include "shortfin/local_device.h"
#include "shortfin/support/stl_extras.h"

namespace shortfin {

class LocalScope;

// Wraps a LocalScope and a DeviceAffinity together. This is used in all
// Scope based APIs as a short-hand for "device" as it contains everything
// needed to do thing with some slice of device queues.
class ScopedDevice {
 public:
  ScopedDevice(LocalScope &scope, DeviceAffinity affinity)
      : scope_(scope), affinity_(affinity) {}
  ScopedDevice(LocalScope &scope, LocalDevice *device)
      : scope_(scope), affinity_(device) {}

  LocalScope &scope() const { return scope_; }
  DeviceAffinity affinity() const { return affinity_; }
  LocalDevice *raw_device() const { return affinity_.device(); }

  std::string to_s() const { return affinity().to_s(); }

  bool operator==(const ScopedDevice &other) const {
    return (&scope_ == &other.scope_) && affinity_ == other.affinity_;
  }

 private:
  LocalScope &scope_;
  DeviceAffinity affinity_;
};

// Handles scheduling state for a scope.
class SHORTFIN_API ScopedScheduler {
 public:
  class SHORTFIN_API Account;

  // Transactions are accumulated into a command buffer by type and in
  // auto-flush mode, the command buffer is submitted upon a change of type.
  enum class TransactionType {
    NONE = 0,
    // Non-aliasing transfer. All transfers submitted in a sequence must not
    // alias each other. If there is a dependency, then they must be flushed
    // before adding an alising transfer.
    TRANSFER = 1,
    // Standalone dispatch that will be enqueued in an individual dispatch.
    SEQUENTIAL_DISPATCH = 2,
    // Parallelizable dispatch that can be enqueued with other parallel
    // dispatches.
    PARALLEL_DISPATCH = 3,
  };

  enum class TransactionMode {
    // All pending command buffers are flushed when the transaction type
    // changes.
    AUTO_FLUSH = 0,
    // Pending command buffers are not flushed until explicitly set to do so.
    EXPLICIT_FLUSH = 1,
  };

  // Control object for a resource that is tracked on the timeline. Each such
  // resource is associated with a single Account. In the case of a resource
  // that is shared across device queues, a consistent account will be chosen
  // as primary (typically corresponding to the lowest numbered queue). This
  // object tracks two things about the resource:
  //   1. Ready: Idle timepoint of the most recent affecting mutation. Since any
  //      such mutation mutation must have been enqueued on a single Account,
  //      we only need to track the timepoint here (against the account's sem).
  //   2. Idle: Fence joined to all timepoints accessing the resource (read and
  //      write). At any given time, this represents the point at which the
  //      resource has no further pending uses.
  // Since TimelineResources are shared (i.e. across subspan storage, etc),
  // they are modeled as reference counted (using non atomics, since this is
  // "scoped" same thread access). They must only be held in a context that
  // is keeping the containing LocalScope alive.
  class SHORTFIN_API TimelineResource {
   public:
    class SHORTFIN_API Ref {
     public:
      Ref() : res_(nullptr) {}
      explicit Ref(TimelineResource *res) : res_(res) { res_->Retain(); }
      Ref(const Ref &other) : res_(other.res_) { res_->Retain(); }
      void operator=(const Ref &other) = delete;
      Ref(Ref &&other) : res_(other.res_) { other.res_ = nullptr; }
      ~Ref() {
        if (res_) res_->Release();
      }
      TimelineResource &operator->() { return *res_; }

     private:
      TimelineResource *res_;
    };
    TimelineResource(TimelineResource &other) = delete;

    Account &account() { return account_; }
    uint64_t &ready_timepoint() { return ready_timepoint_; }
    iree_hal_fence_t *idle_fence() { return idle_fence_.get(); }

   private:
    TimelineResource(Account &account) : account_(account) {}
    static Ref New(Account &account) {
      return Ref(new TimelineResource(account));
    }
    void Retain() { refcnt_++; }
    void Release() {
      if (--refcnt_ == 0) delete this;
    }
    uint64_t ready_timepoint_ = 0;
    int refcnt_ = 0;
    Account &account_;
    iree_hal_fence_ptr idle_fence_;
    friend class Account;
  };

  // Accounting structure for a single logical device (LocalDevice*), which
  // means that each addressable queue gets its own Account.
  class SHORTFIN_API Account {
   public:
    Account(LocalDevice *device);
    LocalDevice *device() const { return device_; }
    iree_hal_device_t *hal_device() { return hal_device_; }
    size_t semaphore_count() const { return 2; }

    // Creates a new TimelineResource associated with this account.
    TimelineResource::Ref NewTimelineResource() {
      return TimelineResource::New(*this);
    }

    // Accesses the active command buffer. This will only be non-null if a
    // pending transaction has been set up (i.e. via AppendCommandBuffer).
    iree_hal_command_buffer_t *active_command_buffer() {
      return active_command_buffer_.get();
    }

   private:
    void Initialize();
    LocalDevice *device_;
    iree_hal_device_t *hal_device_;
    TransactionType active_tx_type_ = TransactionType::NONE;
    iree_hal_command_buffer_ptr active_command_buffer_;
    iree_hal_queue_affinity_t active_queue_affinity_bits_;

    // Timepoint at which this device is considered idle, inclusive of any
    // active_command_buffer that has not yet been submitted. This means
    // that at any time, a wait on this timepoint will produce the expected
    // temporal sequencing, but it may represent a point in the future where
    // the work needed to reach it has not been submitted yet.
    // This means that it has two interpretations based on the state of
    // active_command_buffer_:
    //   - nullptr:  All work has been flushed to reach this timepoint.
    //   - !nullptr: When flushed, the active command buffer will be submitted
    //     to signal this timepoint, fulfilling any waiters.
    // Whenever we transition active_command_buffer_ from nullptr,
    // idle_timepoint_ *must* be incremented by at least 1 (or otherwise
    // an eventual submission would submit a duplicate timepoint). We maintain
    // this invariant by incrementing the global timepoint by 1 for every
    // insertion of a command buffer row(s) and set the idle_timepoint_ to
    // this new value. At most, this results in wasted timepoints (which are
    // free), and it eliminates the hazzard.
    uint64_t idle_timepoint_ = 0;
    iree_hal_semaphore_ptr sem_;
    friend class ScopedScheduler;
  };

  void set_transaction_mode(TransactionMode tx_mode);
  TransactionMode transaction_mode() const { return tx_mode_; }

  // Given a ScopedDevice (which may logically bind to multiple queues),
  // returns a deterministic Account associated with the device that can be
  // used for accounting and scheduling.
  Account &GetDefaultAccount(ScopedDevice &device);

  // Sets up |device| for appending commands to a command buffer, invoking
  // callback to complete the mutation. Depending on the current transaction
  // mode and tx_type, this may involve flushing the current command buffer.
  // This will always allocate a new timepoint from the global timeline and
  // increment the elected device account's idle timepoint. It will also
  // set any affinity bits on the pending submission.
  void AppendCommandBuffer(ScopedDevice &device, TransactionType tx_type,
                           std::function<void(Account &)> callback);

 private:
  void Initialize(std::span<LocalDevice *const> devices);
  // Each distinct hal device gets an account.
  std::vector<Account> accounts_;
  // Accounts indexed by LocalDeviceAddress::device_id().
  std::unordered_map<uint64_t, Account *> accounts_by_device_id_;
  size_t semaphore_count_ = 0;

  // Transaction management.
  TransactionMode tx_mode_ = TransactionMode::AUTO_FLUSH;
  TransactionType current_tx_type_ = TransactionType::NONE;

  friend class LocalScope;
};

// A logical scope of execution, consisting of participating devices,
// resources, and timelines. Most interaction with the compute resources
// is done on these instances.
//
// The scope is generally instantiated with a slice of system resources,
// and produces an arrangement that is easy to use vs maximally diverse.
//
// Devices
// -------
// The scope is initialized with a list of participating devices, which is
// a subset of all devices managed by the LocalSystem. Each device is given
// a logical name of the form `<device_class><index>`, by default using the
// LocalDeviceAddress::logical_device_class as the `<device_class>`. In exotic
// situations, this can be customized. By default, devices are added in the
// order defined by the system and will have an `<index>` corresponding to
// their order. It is up to the constructor to produce a sensible arrangement.
class SHORTFIN_API LocalScope {
 public:
  // Initialize with devices using logical_device_class as the device class.
  LocalScope(std::span<LocalDevice *const> devices);
  // Initialize with devices with custom device class names.
  LocalScope(
      std::span<const std::pair<std::string_view, LocalDevice *>> devices);
  LocalScope(const LocalScope &) = delete;
  // Ensure polymorphic.
  virtual ~LocalScope();

  // Device access.
  // Throws std::invalid_argument on lookup failure.
  LocalDevice *raw_device(std::string_view name) const;
  const std::unordered_map<std::string_view, LocalDevice *> named_devices()
      const {
    return named_devices_;
  }
  LocalDevice *raw_device(int index) const;
  LocalDevice *raw_device(LocalDevice *device) const { return device; }
  const std::vector<LocalDevice *> &raw_devices() const { return devices_; }
  std::vector<std::string_view> device_names() const;

  // Variadic helper for making a DeviceAffinity from any of:
  //  * Explicit LocalDevice*
  //  * Device name (from a LocalScope)
  //  * Device index (from a LocalScope)
  // If at any point during accumulation, the DeviceAffinity would be invalid,
  // then a std::invalid_argument exception is thrown. Any failure to resolve
  // a name or index will also throw a std::invalid_argument.
  ScopedDevice device() { return ScopedDevice(*this, DeviceAffinity()); }
  template <typename T, typename... Args>
  ScopedDevice device(T first, Args... args) {
    return ScopedDevice(
        *this, device(args...).affinity() | DeviceAffinity(raw_device(first)));
  }
  ScopedDevice device(LocalDevice *d) {
    return ScopedDevice(*this, DeviceAffinity(d));
  }
  ScopedScheduler &scheduler() { return scheduler_; }
  ScopedScheduler::TimelineResource::Ref NewTimelineResource(
      ScopedDevice &device) {
    return scheduler().GetDefaultAccount(device).NewTimelineResource();
  }

 private:
  void AddDevice(std::string_view device_class, LocalDevice *device);
  void Initialize();  // Called after all devices are added.

  string_interner interner_;

  // Map of `<device_class>` to the count of that class contained.
  std::unordered_map<std::string_view, int> device_class_count_;
  // Ordered devices.
  std::vector<LocalDevice *> devices_;
  // Map of `<device_class><index>` to LocalDevice.
  std::unordered_map<std::string_view, LocalDevice *> named_devices_;
  ScopedScheduler scheduler_;
};

}  // namespace shortfin

#endif  // SHORTFIN_LOCAL_SCOPE_H
