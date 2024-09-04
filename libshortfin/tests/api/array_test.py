# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import array
import pytest

import shortfin as sf
import shortfin.array as sfnp


@pytest.fixture
def lsys():
    sc = sf.host.CPUSystemBuilder()
    lsys = sc.create_system()
    yield lsys
    lsys.shutdown()


@pytest.fixture
def scope(lsys):
    return lsys.create_scope()


@pytest.fixture
def device(scope):
    return scope.device(0)


def test_storage_constructor(lsys, device):
    async def main():
        s = sfnp.storage.allocate_host(device, 8)
        s.fill(b"\0\1\2\3")
        await device
        ary = sfnp.device_array(s, [2, 4], sfnp.uint8)
        assert ary.dtype == sfnp.uint8
        assert ary.shape == [2, 4]
        assert str(ary) == "{{0, 1, 2, 3},\n {0, 1, 2, 3}}"
        assert ary.device == device
        assert ary.storage == s

    lsys.run(main())


def test_device_constructor(lsys, device):
    async def main():
        ary = sfnp.device_array(device, [2, 4], sfnp.uint8)
        ary.storage.fill(b"\0\1\2\3")
        await device
        assert ary.dtype == sfnp.uint8
        assert ary.shape == [2, 4]
        assert str(ary) == "{{0, 1, 2, 3},\n {0, 1, 2, 3}}"
        assert ary.device == device

    lsys.run(main())


def test_fill_copy_from_for_transfer(lsys, device):
    async def main():
        src = sfnp.device_array(device, [2, 4], sfnp.uint8)
        src.fill(b"\0\1\2\3")
        dst = src.for_transfer()
        dst.copy_from(src)
        await device
        assert str(dst) == "{{0, 1, 2, 3},\n {0, 1, 2, 3}}"

    lsys.run(main())


def test_fill_copy_to_for_transfer(lsys, device):
    async def main():
        src = sfnp.device_array(device, [2, 4], sfnp.uint8)
        src.fill(b"\0\1\2\3")
        dst = src.for_transfer()
        src.copy_to(dst)
        await device
        assert str(dst) == "{{0, 1, 2, 3},\n {0, 1, 2, 3}}"

    lsys.run(main())


def test_shape_overflow(lsys, device):
    async def main():
        s = sfnp.storage.allocate_host(device, 4)
        _ = sfnp.device_array(s, [2, 4], sfnp.uint8)

    with pytest.raises(
        ValueError, match="Array storage requires at least 8 bytes but has only 4"
    ):
        lsys.run(main())


@pytest.mark.parametrize(
    "dtype,code,py_value,expected_str",
    [
        (sfnp.int8, "b", 42, "{{42, 42, 42, 42},\n {42, 42, 42, 42}}"),
        (sfnp.int16, "h", 42, "{{42, 42, 42, 42},\n {42, 42, 42, 42}}"),
        (sfnp.int32, "i", 42, "{{42, 42, 42, 42},\n {42, 42, 42, 42}}"),
        (
            sfnp.float32,
            "f",
            42.0,
            "{{ 42.,  42.,  42.,  42.},\n { 42.,  42.,  42.,  42.}}",
        ),
        (
            sfnp.float64,
            "d",
            42.0,
            "{{ 42.,  42.,  42.,  42.},\n { 42.,  42.,  42.,  42.}}",
        ),
    ],
)
def test_xtensor_types(scope, dtype, code, py_value, expected_str):
    ary = sfnp.device_array.for_host(scope.device(0), [2, 4], dtype)
    ary.storage.data = array.array(code, [py_value] * 8)
    s = str(ary)
    print("__str__ =", s)
    assert expected_str == s, f"Expected '{expected_str}' == '{s}'"
    r = repr(ary)
    print("__repr__ =", r)
    assert expected_str in r, f"Expected '{expected_str}' in '{r}'"
