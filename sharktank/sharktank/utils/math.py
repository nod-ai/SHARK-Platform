# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from numbers import Number


def ceildiv(a: int | float, b: int | float) -> int | float:
    return -(a // -b)


def round_up_to_multiple_of(x: Number, multiple: Number) -> Number:
    return x + (-x % multiple)
