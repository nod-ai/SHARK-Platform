# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from _shortfin import lib as sfl


@pytest.fixture
def lsys():
    sc = sfl.host.CPUSystemBuilder()
    return sc.create_local_system()


@pytest.fixture
def scope(lsys):
    return lsys.create_scope()


def test_storage(scope):
    s = sfl.array.storage.allocate_host(scope.device("cpu0"), 32)
    print(s)
