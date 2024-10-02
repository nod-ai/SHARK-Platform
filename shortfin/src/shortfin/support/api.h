// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_SUPPORT_API_H
#define SHORTFIN_SUPPORT_API_H

// Note that in the case of transitive intra-dylib deps, both BUILDING
// and USING can be defined. In this case, BUILDING takes precedence and
// is the correct choice.
#if defined(_SHORTFIN_BUILDING_DYLIB)
#if (defined(_WIN32) || defined(__CYGWIN__))
#define SHORTFIN_API __declspec(dllexport)
#else
#define SHORTFIN_API __attribute__((visibility("default")))
#endif
#elif defined(_SHORTFIN_USING_DYLIB)
#if (defined(_WIN32) || defined(__CYGWIN__))
#define SHORTFIN_API __declspec(dllimport)
#else
#define SHORTFIN_API
#endif
#else
#define SHORTFIN_API
#endif

#endif  // SHORTFIN_SUPPORT_API_H
