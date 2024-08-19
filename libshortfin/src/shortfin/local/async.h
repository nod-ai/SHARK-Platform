// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_ASYNC_H
#define SHORTFIN_LOCAL_ASYNC_H

#include <any>
#include <functional>

#include "shortfin/support/api.h"
#include "shortfin/support/iree_concurrency.h"
#include "shortfin/support/iree_helpers.h"

namespace shortfin::local {

// CompletionEvents are the most basic form of awaitable object. They
// encapsulate a native iree_wait_source_t (which multiplexes any supported
// system level wait primitive) with a resource baton which keeps any needed
// references alive for the duration of all copies.
//
// Depending on the system wait source used, there may be a limited exception
// side-band (i.e. a way to signal that the wait handle has failed and have
// that propagate to consumers). However, in general, this is a very coarse
// mechanism. For rich result and error propagation, see the higher level
// Promise/Future types, which can be signalled with either a result or
// exception.
class SHORTFIN_API CompletionEvent {
 public:
  CompletionEvent();
  CompletionEvent(iree::shared_event::ref event);
  CompletionEvent(iree::hal_semaphore_ptr sem, uint64_t payload);
  CompletionEvent(CompletionEvent &&other)
      : wait_source_(other.wait_source_),
        resource_baton_(std::move(other.resource_baton_)) {
    other.wait_source_ = iree_wait_source_immediate();
  }
  CompletionEvent(const CompletionEvent &other)
      : wait_source_(other.wait_source_),
        resource_baton_(other.resource_baton_) {}
  CompletionEvent &operator=(const CompletionEvent &other) {
    wait_source_ = other.wait_source_;
    resource_baton_ = other.resource_baton_;
    return *this;
  }
  ~CompletionEvent();

  // Returns true if this CompletionEvent is ready.
  bool is_ready();
  // Block the current thread for up to |timeout|. If a non-infinite timeout
  // was given and the timeout expires while waiting, returns false. In all
  // other cases, returns true.
  // This should not be used in worker loops.
  bool BlockingWait(iree_timeout_t timeout = iree_infinite_timeout());

  // Access the raw wait source.
  const iree_wait_source_t &wait_source() { return wait_source_; }

 private:
  iree_wait_source_t wait_source_;
  // A baton used to keep any needed backing resource alive.
  std::any resource_baton_;
};

}  // namespace shortfin::local

#endif  // SHORTFIN_LOCAL_ASYNC_H
