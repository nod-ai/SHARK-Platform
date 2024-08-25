# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import array
import pytest
import time

import shortfin as sf


@pytest.fixture
def lsys():
    sc = sf.host.CPUSystemBuilder()
    lsys = sc.create_system()
    yield lsys
    lsys.shutdown()


@pytest.fixture
def scope(lsys):
    # TODO: Should adopt the main thread.
    worker = lsys.create_worker("main")
    return lsys.create_scope(worker)


def test_storage(scope):
    storage = sf.array.storage.allocate_host(scope.device(0), 32)
    print(storage)
    ary = sf.array.device_array(storage, [2, 4], sf.array.float32)
    print(ary)
    print(ary.shape)
    assert ary.shape == [2, 4]
    assert ary.dtype == sf.array.float32

    print("ARY.DEVICE=", ary.device, ary.device.__class__)
    print("SCOPE.DEVICE=", scope.device(0))
    print("EQ:", ary.device == scope.device(0))

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


@pytest.mark.parametrize(
    "dtype,code,py_value,expected_repr",
    [
        (sf.array.int8, "b", 42, "{{42, 42, 42, 42},\n {42, 42, 42, 42}}"),
        (sf.array.int16, "h", 42, "{{42, 42, 42, 42},\n {42, 42, 42, 42}}"),
        (sf.array.int32, "i", 42, "{{42, 42, 42, 42},\n {42, 42, 42, 42}}"),
        (
            sf.array.float32,
            "f",
            42.0,
            "{{ 42.,  42.,  42.,  42.},\n { 42.,  42.,  42.,  42.}}",
        ),
        (
            sf.array.float64,
            "d",
            42.0,
            "{{ 42.,  42.,  42.,  42.},\n { 42.,  42.,  42.,  42.}}",
        ),
    ],
)
def test_xtensor_types(scope, dtype, code, py_value, expected_repr):
    ary = sf.array.device_array.for_host(scope.device(0), [2, 4], dtype)
    ary.storage.data = array.array(code, [py_value] * 8)
    r = repr(ary)
    print(r)
    assert expected_repr in r, f"Expected '{expected_repr}' in '{r}'"


def test_device_array(scope):
    ary1 = sf.array.device_array(scope.device(0), [32, 1, 4], sf.array.float32)
    print(ary1)
    assert ary1.shape == [32, 1, 4]
    assert ary1.dtype == sf.array.float32
    assert scope.device(0) == ary1.device

    hary1 = sf.array.device_array.for_transfer(ary1)
    print(hary1)
    assert isinstance(hary1, sf.array.device_array)
    assert hary1.shape == ary1.shape
    assert hary1.dtype == ary1.dtype
    assert hary1.device == ary1.device


def test_device_array_fill(scope):
    ary1 = sf.array.device_array(scope.device(0), [32, 1, 4], sf.array.int32)
    ary1.storage.fill(array.array("i", [42]))
    # TODO: Transfer to host and verify.
