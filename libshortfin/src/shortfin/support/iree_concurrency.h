// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// C++ helpers for using IREE threading primitives.

#ifndef SHORTFIN_SUPPORT_IREE_THREADING_H
#define SHORTFIN_SUPPORT_IREE_THREADING_H

#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/threading.h"
#include "iree/base/internal/wait_handle.h"
#include "shortfin/support/iree_helpers.h"

namespace shortfin::iree {

SHORTFIN_IREE_DEF_PTR(thread);

// Wraps an iree::slim_mutex as an RAII object.
class slim_mutex {
 public:
  slim_mutex() { iree_slim_mutex_initialize(&mu_); }
  slim_mutex(const slim_mutex &) = delete;
  slim_mutex &operator=(const slim_mutex &) = delete;
  ~slim_mutex() { iree_slim_mutex_deinitialize(&mu_); }

  operator iree_slim_mutex_t *() { return &mu_; }

 private:
  iree_slim_mutex_t mu_;
};

// RAII slim mutex lock guard.
class slim_mutex_lock_guard {
 public:
  slim_mutex_lock_guard(slim_mutex &mu) : mu_(mu) { iree_slim_mutex_lock(mu_); }
  ~slim_mutex_lock_guard() { iree_slim_mutex_unlock(mu_); }

 private:
  slim_mutex &mu_;
};

// Wrapper around an iree::event_t.
class event {
 public:
  event(bool initial_state) {
    SHORTFIN_THROW_IF_ERROR(iree_event_initialize(initial_state, &event_));
  }
  event(const event &) = delete;
  event &operator=(const event &) = delete;
  ~event() { iree_event_deinitialize(&event_); }

  void set() { iree_event_set(&event_); }
  void reset() { iree_event_reset(&event_); }

  iree_wait_source_t await() { return iree_event_await(&event_); }

 private:
  iree_event_t event_;
};

// An event that is ref-counted.
class shared_event : private event {
 public:
  class ref {
   public:
    ref() = default;
    explicit ref(bool initial_state) : inst_(new shared_event(initial_state)) {}
    ref(const ref &other) : inst_(other.inst_) {
      if (inst_) {
        inst_->ref_count_.fetch_add(1);
      }
    }
    ref &operator=(const ref &other) {
      if (inst_ != other.inst_) {
        reset();
        inst_ = other.inst_;
        if (inst_) {
          inst_->ref_count_.fetch_add(1);
        }
      }
      return *this;
    }
    ref(ref &&other) : inst_(other.inst_) { other.inst_ = nullptr; }
    ~ref() { reset(); }

    operator bool() const { return inst_ != nullptr; }
    iree::event *operator->() const { return inst_; }
    void reset() {
      if (inst_) {
        manual_release();
        inst_ = nullptr;
      }
    }

    int ref_count() const { return inst_->ref_count_.load(); }

    // Manually retain the event. Must be matched by a call to release().
    void manual_retain() { inst_->ref_count_.fetch_add(1); }
    void manual_release() {
      if (inst_->ref_count_.fetch_sub(1) == 1) {
        delete inst_;
      }
    }

   private:
    explicit ref(iree::shared_event *inst) : inst_(inst) {}
    shared_event *inst_ = nullptr;
    friend class iree::shared_event;
  };

  static ref create(bool initial_state) {
    return ref(new shared_event(initial_state));
  }

 private:
  using event::event;
  ~shared_event() = default;

  std::atomic<int> ref_count_{1};
};

}  // namespace shortfin::iree

#endif  // SHORTFIN_SUPPORT_IREE_THREADING_H
