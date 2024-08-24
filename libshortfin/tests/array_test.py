# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import array
import pytest
import time

from _shortfin import lib as sfl


@pytest.fixture
def lsys():
    sc = sfl.local.host.CPUSystemBuilder()
    lsys = sc.create_system()
    yield lsys
    lsys.shutdown()


@pytest.fixture
def scope(lsys):
    # TODO: Should adopt the main thread.
    worker = lsys.create_worker("main")
    return lsys.create_scope(worker)


def test_storage(scope):
    storage = sfl.array.storage.allocate_host(scope.device(0), 32)
    print(storage)
    ary = sfl.array.device_array(storage, [2, 4], sfl.array.float32)
    print(ary)
    print(ary.shape)
    assert ary.shape == [2, 4]
    assert ary.dtype == sfl.array.float32
    assert ary.device == scope.device(0)

    # Mapping API contract.
    with storage.map(read=True) as m:
        assert m.valid
        mv = memoryview(m)
        assert len(mv) == 32
    assert not m.valid

    storage.data = array.array("f", [1.234534523] * 8)
    print("WRITTEN:", ary)

    read_back = array.array("f")
    read_back.frombytes(storage.data)
    print("READ BACK:", read_back)


def test_device_array(scope):
    ary1 = sfl.array.device_array(scope.device(0), [32, 1, 4], sfl.array.float32)
    print(ary1)
    assert ary1.shape == [32, 1, 4]
    assert ary1.dtype == sfl.array.float32
    assert scope.device(0) == ary1.device

    hary1 = sfl.array.device_array.for_transfer(ary1)
    print(hary1)
    assert isinstance(hary1, sfl.array.device_array)
    assert hary1.shape == ary1.shape
    assert hary1.dtype == ary1.dtype
    assert hary1.device == ary1.device


def test_device_array_fill(scope):
    ary1 = sfl.array.device_array(scope.device(0), [32, 1, 4], sfl.array.float32)
    ary1.storage.fill(array.array("f", [1.0]))
    # TODO: Transfer to host and verify.
