// Copyright 2024 Advanced Micro Devices, Inc
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

namespace shortfin {

namespace iree_type_detail {
struct iree_thread_ptr_helper {
  static void retain(iree_thread_t *obj) { iree_thread_retain(obj); }
  static void release(iree_thread_t *obj) { iree_thread_release(obj); }
};
};  // namespace iree_type_detail

using iree_thread_ptr =
    iree_object_ptr<iree_thread_t, iree_type_detail::iree_thread_ptr_helper>;

// Wraps an iree_slim_mutex_t as an RAII object.
class iree_slim_mutex {
 public:
  iree_slim_mutex() { iree_slim_mutex_initialize(&mu_); }
  ~iree_slim_mutex() { iree_slim_mutex_deinitialize(&mu_); }

  operator iree_slim_mutex_t *() { return &mu_; }

 private:
  iree_slim_mutex_t mu_;
};

// RAII slim mutex lock guard.
class iree_slim_mutex_lock_guard {
 public:
  iree_slim_mutex_lock_guard(iree_slim_mutex &mu) : mu_(mu) {
    iree_slim_mutex_lock(mu_);
  }
  ~iree_slim_mutex_lock_guard() { iree_slim_mutex_unlock(mu_); }

 private:
  iree_slim_mutex mu_;
};

// Wrapper around an iree_event_t.
class iree_event {
 public:
  iree_event(bool initial_state) {
    SHORTFIN_THROW_IF_ERROR(iree_event_initialize(initial_state, &event_));
  }
  ~iree_event() { iree_event_deinitialize(&event_); }

  void set() { iree_event_set(&event_); }
  void reset() { iree_event_reset(&event_); }

  iree_wait_source_t await() { return iree_event_await(&event_); }

 private:
  iree_event_t event_;
};

}  // namespace shortfin

#endif  // SHORTFIN_SUPPORT_IREE_THREADING_H
