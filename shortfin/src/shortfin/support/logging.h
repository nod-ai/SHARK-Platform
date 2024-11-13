// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_SUPPORT_LOGGING_H
#define SHORTFIN_SUPPORT_LOGGING_H

#include "iree/base/tracing.h"
#include "shortfin/support/api.h"
#include "spdlog/spdlog.h"

#if !defined(SHORTFIN_LOG_LIFETIMES)
#define SHORTFIN_LOG_LIFETIMES 0
#endif

// Scheduler logging.
#define SHORTFIN_SCHED_LOG_ENABLED 0
#if SHORTFIN_SCHED_LOG_ENABLED
#define SHORTFIN_SCHED_LOG(...) shortfin::logging::info("SCHED: " __VA_ARGS__)
#else
#define SHORTFIN_SCHED_LOG(...)
#endif

// Tracing macros. These are currently just aliases of the underlying IREE
// macros, but we maintain the ability to redirect them in the future (i.e.
// for certain kinds of library builds, etc).
#define SHORTFIN_TRACE_SCOPE IREE_TRACE_SCOPE
#define SHORTFIN_TRACE_SCOPE_NAMED(name_literal) \
  IREE_TRACE_SCOPE_NAMED(name_literal)
#define SHORTFIN_TRACE_SCOPE_ID IREE_TRACE_SCOPE_ID

namespace shortfin::logging {

SHORTFIN_API void InitializeFromEnv();

// TODO: Re-export doesn't really work like this. Need to define API
// exported trampolines for cross library use.
using spdlog::debug;
using spdlog::error;
using spdlog::info;
using spdlog::warn;

#if SHORTFIN_LOG_LIFETIMES
template <typename T>
inline void construct(const char* type_name, T* inst) {
  info("new {}({})", type_name, static_cast<void*>(inst));
}
template <typename T>
inline void destruct(const char* type_name, T* inst) {
  info("delete {}({})", type_name, static_cast<void*>(inst));
}
#else
template <typename T>
inline void construct(const char *type_name, T *) {}
template <typename T>
inline void destruct(const char *type_name, T *) {}
#endif

}  // namespace shortfin::logging

#endif  // SHORTFIN_SUPPORT_LOGGING_H
