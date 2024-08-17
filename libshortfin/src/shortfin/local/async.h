// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_ASYNC_H
#define SHORTFIN_LOCAL_ASYNC_H

#include <functional>

#include "shortfin/support/api.h"
#include "shortfin/support/iree_concurrency.h"
#include "shortfin/support/iree_helpers.h"

namespace shortfin::local {

// Encapsulates an IREE wait source and combines it with additional plumbing
// to support resource management. On move, the source of the move is reset
// to an immediate wait source and it's resource controller is cleared.
class SHORTFIN_API SingleWaitFuture {
 public:
  // The SingleWaitFuture can contain a ResourceControl function. If present,
  // then it is called to retain/release a backing resource.
  enum class ResourceCommand {
    RETAIN,
    RELEASE,
  };
  using ResourceControl = std::function<void(ResourceCommand)>;

  SingleWaitFuture();
  SingleWaitFuture(iree_shared_event::ref event);
  SingleWaitFuture(iree_hal_semaphore_ptr sem, uint64_t payload);
  SingleWaitFuture(SingleWaitFuture &&other)
      : wait_source_(other.wait_source_),
        resource_control_(std::move(other.resource_control_)) {
    other.wait_source_ = iree_wait_source_immediate();
  }
  SingleWaitFuture(const SingleWaitFuture &other)
      : wait_source_(other.wait_source_),
        resource_control_(other.resource_control_) {
    resource_control_(ResourceCommand::RETAIN);
  }
  SingleWaitFuture &operator=(const SingleWaitFuture &other) {
    if (other.resource_control_) {
      other.resource_control_(ResourceCommand::RETAIN);
    }
    if (resource_control_) {
      resource_control_(ResourceCommand::RELEASE);
    }
    wait_source_ = other.wait_source_;
    resource_control_ = other.resource_control_;
    return *this;
  }
  ~SingleWaitFuture();

  // Returns true if this SingleWaitFuture is ready.
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
  ResourceControl resource_control_;
};

}  // namespace shortfin::local

#endif  // SHORTFIN_LOCAL_ASYNC_H
