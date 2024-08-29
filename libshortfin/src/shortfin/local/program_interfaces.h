// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Standalone interfaces needed for marshaling as part of a ProgramInvocation.h.
// They are available in this dep-free header in order to ease the burden on
// types that would otherwise need to pull in all of the includes.

#ifndef SHORTFIN_LOCAL_PROGRAM_INTERFACES_H
#define SHORTFIN_LOCAL_PROGRAM_INTERFACES_H

#include "shortfin/support/api.h"

namespace shortfin::local {

class SHORTFIN_API ProgramInvocation;

// The type of barrier that should be managed for a program resource.
enum class ProgramResourceBarrier {
  // The caller has explicitly not stated a preference.
  DEFAULT,

  // The argument will be used by the program for input and the program
  // must not perform operations on it until all pending mutations have
  // been completed. Concurrent reads/uses are permitted.
  // This is the default concurrency in most situations.
  READ,

  // The argument will be used for input/output and the program must not
  // perform operations on it until all prior mutations and uses have been
  // complete.
  WRITE,

  // No concurrency barriers will be emplaced on behalf of the argument,
  // explicitly allowing racy access. The program and the caller must
  // ensure that only valid accesses are made.
  NONE,
};

// Implemented by a class if it can marshal itself to an invocation as an
// argument.
class SHORTFIN_API ProgramInvocationMarshalable {
 public:
  // Adds this object as an invocation argument.
  virtual void AddAsInvocationArgument(ProgramInvocation *inv,
                                       ProgramResourceBarrier barrier) = 0;
};

}  // namespace shortfin::local

#endif  // SHORTFIN_LOCAL_PROGRAM_INTERFACES_H
