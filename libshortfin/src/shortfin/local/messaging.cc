// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local/messaging.h"

#include "shortfin/support/logging.h"

namespace shortfin::local {

Message::~Message() {
  logging::info("MESSAGE DTOR: {}", reinterpret_cast<void*>(owner_.Control));
}

std::string Queue::to_s() const {
  return fmt::format("Queue(name={})", options().name);
}

}  // namespace shortfin::local
