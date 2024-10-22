# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import array
import logging
import math
import numpy as np
import pytest

import shortfin as sf
import shortfin.array as sfnp
import shortfin.debug.array as debug_array
import shortfin.host


@pytest.fixture
def lsys():
    sc = sf.host.CPUSystemBuilder()
    lsys = sc.create_system()
    yield lsys
    lsys.shutdown()


@pytest.fixture
def fiber(lsys):
    return lsys.create_fiber()


@pytest.fixture
def device(fiber):
    return fiber.device(0)


# A fixture to capture the logging output
@pytest.fixture
def caplog(caplog):
    caplog.set_level(logging.DEBUG)
    return caplog


def test_to_np_from_device_array(device, lsys):
    async def main():
        shape = [4, 16, 128]
        src = sfnp.device_array(device, shape, dtype=sfnp.float32)
        data = [i for i in range(math.prod([1, 16, 128]))]
        for i in range(4):
            src.view(i).items = data
            data.reverse()

        # Convert to np array
        np_array = await debug_array.to_np(src)

        # Check array is numpy array
        assert isinstance(np_array, np.ndarray)

        # Check if the shape is correct
        assert np_array.shape == tuple(shape)

        # Check if the dtype is correct
        assert np_array.dtype == np.float32

        # Check if the values are correct
        for i in range(4):
            assert np_array[i].flatten().tolist() == src.view(i).items.tolist()

    lsys.run(main())


def test_to_np_from_device_f16(device, lsys):
    def int_to_f16_uint(n):
        """Converts an integer to a float16 uint, for convenient testing."""
        if n == 0:
            return 0
        exponent = n.bit_length() - 1
        if n == 2**exponent:
            return (exponent + 15) << 10
        else:
            fraction = n - 2**exponent
            fraction_bits = fraction << (10 - exponent)
            return ((exponent + 15) << 10) | fraction_bits

    async def main():
        shape = [4, 42, 48]
        src = sfnp.device_array(device, shape, dtype=sfnp.float16)
        data = [int_to_f16_uint(i) for i in range(math.prod([1, 42, 48]))]
        for i in range(4):
            src.view(i).items = data
            data.reverse()

        # Convert to np array
        np_array = await debug_array.to_np(src)

        # Check array is numpy array
        assert isinstance(np_array, np.ndarray)

        # Check if the shape is correct
        assert np_array.shape == tuple(shape)

        # Check if the dtype is correct
        assert np_array.dtype == np.float16

        compare_to = np.zeros([4, 42, 48], dtype=np.float16)
        data = [i for i in range(math.prod([1, 42, 48]))]
        for i in range(4):
            compare_to[i] = np.array(data).reshape(42, 48)
            data.reverse()

        # Check if the values are correct
        for i in range(4):
            assert np_array[i].flatten().tolist() == compare_to[i].flatten().tolist()

    lsys.run(main())


def test_dump_array(device, lsys, caplog):
    async def main():
        shape = [4, 16, 128]
        src = sfnp.device_array(device, shape, dtype=sfnp.float32)
        data = [float(i) for i in range(math.prod([1, 16, 128]))]
        for i in range(4):
            src.view(i).items = data
            data.reverse()

        # Ensure array is dumped properly to log output
        await debug_array.dump_array(src)
        src_np_array = await debug_array.to_np(src)
        assert str(src_np_array) in caplog.text

    lsys.run(main())


def test_fill_array(device, lsys):
    async def main():
        shape = [4, 16, 128]
        src = sfnp.device_array(device, shape, dtype=sfnp.float32)
        data = [0 for _ in range(math.prod([1, 16, 128]))]
        for i in range(4):
            src.view(i).items = data

        # Fill array
        fill_value = 3.14
        np_array = await debug_array.fill_array(src, fill_value)

        # Check if the values are correct
        compare_to = np.zeros([16, 128], dtype=np.float32)
        compare_to.fill(fill_value)
        for i in range(4):
            assert np_array[i].tolist() == compare_to.tolist()

    lsys.run(main())


def test_find_mode_basic():
    arr = np.array([1, 2, 3, 3, 4, 5, 5, 5, 5, 5])
    mode, count = debug_array.find_mode(arr)
    assert mode == 5
    assert count == 5


def test_find_mode_empty():
    arr = np.array([])
    mode, count = debug_array.find_mode(arr)
    assert math.isnan(mode)
    assert count == 0


def test_find_mode_multi_dim():
    arr = np.array([[1, 2, 3], [3, 4, 5], [5, 5, 5]])
    mode, count = debug_array.find_mode(arr, axis=1)
    assert mode.tolist() == [1, 3, 5]
    assert count.tolist() == [1, 1, 3]


def test_find_mode_keep_dim():
    arr = np.array([[1, 2, 3], [3, 4, 5], [5, 5, 5]])
    mode, count = debug_array.find_mode(arr, axis=1, keepdims=True)
    assert mode.tolist() == [[1], [3], [5]]
    assert count.tolist() == [[1], [1], [3]]


def test_log_tensor_stats_basic(device, lsys, caplog):
    async def main():
        shape = [1, 6]
        src = sfnp.device_array(device, shape, dtype=sfnp.float32)
        data = [1, 2, 3, 3, 4, 5]
        src.view(0).items = data

        # Ensure array stats are logged properly
        await debug_array.log_tensor_stats(src)
        assert "NaN count: 0 / 6" in caplog.text
        assert "Shape: (1, 6), dtype: float32" in caplog.text
        assert "Min (excluding NaN): 1.0" in caplog.text
        assert "Max (excluding NaN): 5.0" in caplog.text
        assert "Mean (excluding NaN): 3.0" in caplog.text
        assert "Mode (excluding NaN): 3" in caplog.text
        assert "First 10 elements: [1. 2. 3. 3. 4. 5.]" in caplog.text
        assert "Last 10 elements: [1. 2. 3. 3. 4. 5.]" in caplog.text

    lsys.run(main())


def test_log_tensor_stats_with_nan(device, lsys, caplog):
    async def main():
        shape = [1, 8]
        src = sfnp.device_array(device, shape, dtype=sfnp.float32)
        data = [3, np.nan, 4, 3, 1, np.nan, 5, 9]
        src.view(0).items = data

        # Ensure array stats are logged properly
        await debug_array.log_tensor_stats(src)
        assert "NaN count: 2 / 8" in caplog.text
        assert "Shape: (1, 8), dtype: float32" in caplog.text
        assert "Min (excluding NaN): 1.0" in caplog.text
        assert "Max (excluding NaN): 9.0" in caplog.text
        assert "Mean (excluding NaN): 4.1666665" in caplog.text
        assert "Mode (excluding NaN): 3" in caplog.text
        assert "First 10 elements: [3. 4. 3. 1. 5. 9.]" in caplog.text
        assert "Last 10 elements: [3. 4. 3. 1. 5. 9.]" in caplog.text

    lsys.run(main())


def test_log_tensor_stats_empty(device, lsys, caplog):
    async def main():
        shape = [1, 0]
        src = sfnp.device_array(device, shape, dtype=sfnp.float32)

        # Ensure array stats are logged properly
        await debug_array.log_tensor_stats(src)
        assert "NaN count: 0 / 0" in caplog.text
        assert "Shape: (1, 0), dtype: float32" in caplog.text

    lsys.run(main())


def test_log_tensor_stats_multi_dim(device, lsys, caplog):
    async def main():
        shape = [3, 3]
        src = sfnp.device_array(device, shape, dtype=sfnp.float32)
        data = [[1, np.nan, 3], [np.nan, 4, 5], [5, np.nan, 7]]
        for i in range(3):
            src.view(i).items = data[i]

        # Ensure array stats are logged properly
        await debug_array.log_tensor_stats(src)
        assert "NaN count: 3 / 9" in caplog.text
        assert "Shape: (3, 3), dtype: float32" in caplog.text
        assert "Min (excluding NaN): 1.0" in caplog.text
        assert "Max (excluding NaN): 7.0" in caplog.text
        assert "Mean (excluding NaN): 4.1666665" in caplog.text
        assert "Mode (excluding NaN): 5" in caplog.text
        assert "First 10 elements: [1. 3. 4. 5. 5. 7.]" in caplog.text
        assert "Last 10 elements: [1. 3. 4. 5. 5. 7.]" in caplog.text

    lsys.run(main())
