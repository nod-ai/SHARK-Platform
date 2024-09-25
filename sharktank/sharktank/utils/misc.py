# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, List
from collections.abc import Iterable
from itertools import zip_longest


def longest_equal_range(l1: List[Any], l2: List[Any]) -> int:
    """Find the longest range that is the same from the start of both lists.
    Returns the greatest `i` such that `l1[0:i] == l2[0:i]`."""
    for i, (a, b) in enumerate(zip(l1, l2)):
        if a != b:
            return i
    return len(zip(l1, l2))


def iterables_equal(iterable1: Iterable, iterable2: Iterable) -> bool:
    return all(v1 == v2 for v1, v2 in zip_longest(iterable1, iterable2))
