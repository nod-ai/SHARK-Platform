// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local/async.h"

namespace shortfin::local {

CompletionEvent::CompletionEvent()
    : wait_source_(iree_wait_source_immediate()) {}

CompletionEvent::CompletionEvent(iree::shared_event::ref event)
    : wait_source_(event->await()) {
  // // It is sufficient to simply capture the event ref in the lambda. It
  // // will be lifetime extended for as long as the resource control
  // // or its copies are alive.
  // resource_control_ = [event](ResourceCommand) {};
}

CompletionEvent::CompletionEvent(iree::hal_semaphore_ptr sem, uint64_t payload)
    : wait_source_(iree_hal_semaphore_await(sem, payload)) {
  // resource_control_ = [sem](ResourceCommand) {};
}

CompletionEvent::~CompletionEvent() {}

bool CompletionEvent::is_ready() {
  iree_status_code_t status_code;
  SHORTFIN_THROW_IF_ERROR(iree_wait_source_query(wait_source_, &status_code));
  return status_code == IREE_STATUS_OK;
}

bool CompletionEvent::BlockingWait(iree_timeout_t timeout) {
  auto status = iree_wait_source_wait_one(wait_source_, timeout);
  if (iree_status_is_deadline_exceeded(status)) {
    iree_status_ignore(status);
    return false;
  }
  SHORTFIN_THROW_IF_ERROR(status);
  return true;
}

}  // namespace shortfin::local
