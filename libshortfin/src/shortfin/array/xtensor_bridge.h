// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_ARRAY_XTENSOR_BRIDGE_H
#define SHORTFIN_ARRAY_XTENSOR_BRIDGE_H

#include <span>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

#include "shortfin/array/array.h"
#include "shortfin/array/storage.h"

namespace shortfin::array {

template <typename EC>
using xt_adaptor =
    xt::xarray_adaptor<typed_mapping<EC>, xt::layout_type::row_major,
                       std::span<const size_t>>;

using float32_xt_adaptor = xt_adaptor<float>;

}  // namespace shortfin::array

#endif  // SHORTFIN_ARRAY_XTENSOR_BRIDGE_H
