# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import logging
import math
import os
import pytest
import sys
from unittest.mock import patch

import shortfin as sf
import shortfin.array as sfnp
from shortfin.array import nputils
import shortfin.host

np = pytest.importorskip("numpy", reason="numpy is not installed")


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


@pytest.fixture(scope="function")
def configure_caplog(caplog):
    caplog.set_level(logging.INFO, logger=None)
    yield caplog


def test_to_np_from_device_array(device):
    def _verify_array(np_arr, compare_to, shape, dtype):
        assert isinstance(np_arr, np.ndarray)
        assert np_arr.shape == shape
        assert np_arr.dtype == dtype
        assert np.array_equal(np_arr, compare_to)

    shape = [4, 16, 128]

    # Test various dtypes (f32, f64, i32, i64)
    src_f32 = sfnp.device_array(device, shape, dtype=sfnp.float32)
    src_f64 = sfnp.device_array(device, shape, dtype=sfnp.float64)
    src_i32 = sfnp.device_array(device, shape, dtype=sfnp.int32)
    src_i64 = sfnp.device_array(device, shape, dtype=sfnp.int64)
    compare_to = np.zeros([4, 16, 128], dtype=np.float32)
    data = [i for i in range(math.prod([1, 16, 128]))]
    for i in range(4):
        src_f32.view(i).items = data
        src_f64.view(i).items = data
        src_i32.view(i).items = data
        src_i64.view(i).items = data
        compare_to[i] = np.array(data).reshape(16, 128)
        data.reverse()

    # Convert to np array
    np_array_f32 = np.array(src_f32)
    np_array_f64 = np.array(src_f64)
    np_array_i32 = np.array(src_i32)
    np_array_i64 = np.array(src_i64)

    _verify_array(np_array_f32, compare_to, tuple(shape), np.float32)
    _verify_array(np_array_f64, compare_to, tuple(shape), np.float64)
    _verify_array(np_array_i32, compare_to, tuple(shape), np.int32)
    _verify_array(np_array_i64, compare_to, tuple(shape), np.int64)


def test_to_np_from_device_f16(device, lsys):
    def _int_to_f16_uint(n):
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

    def _verify_array(np_arr, compare_to, shape, dtype):
        assert isinstance(np_arr, np.ndarray)
        assert np_arr.shape == shape
        assert np_arr.dtype == dtype
        assert np.array_equal(np_arr, compare_to)

    shape = [4, 42, 48]
    src = sfnp.device_array(device, shape, dtype=sfnp.float16)
    compare_to = np.zeros([4, 42, 48], dtype=np.float16)
    data_uint = [_int_to_f16_uint(i) for i in range(math.prod([1, 42, 48]))]
    data = [i for i in range(math.prod([1, 42, 48]))]
    for i in range(4):
        src.view(i).items = data_uint
        compare_to[i] = np.array(data, dtype=np.float16).reshape(42, 48)
        data.reverse()
        data_uint.reverse()

    # Convert to np array
    np_array = np.array(src)
    _verify_array(np_array, compare_to, tuple(shape), np.float16)


def test_dump_array(device):
    shape = [4, 16, 128]
    src = sfnp.device_array(device, shape, dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod([1, 16, 128]))]
    for i in range(4):
        src.view(i).items = data
        data.reverse()

    # Ensure array is dumped properly to log output
    log_messages = []
    with patch.object(
        nputils.logger,
        "debug",
        side_effect=lambda message: log_messages.append(message),
    ):
        nputils.debug_dump_array(src)
        src_np_array = np.array(src)
        arr_str = str(src_np_array)
        assert arr_str == str(log_messages[0])


def test_fill_array(device, lsys):
    shape = [4, 16, 128]
    src = sfnp.device_array(device, shape, dtype=sfnp.float32)
    data = [0 for _ in range(math.prod([1, 16, 128]))]
    for i in range(4):
        src.view(i).items = data

    # Fill array
    fill_value = 3.14
    np_array = nputils.debug_fill_array(src, fill_value)

    # Check if the values are correct
    compare_to = np.zeros([16, 128], dtype=np.float32)
    compare_to.fill(fill_value)
    for i in range(4):
        assert np_array[i].tolist() == compare_to.tolist()


def test__find_mode_basic():
    arr = np.array([1, 2, 3, 3, 4, 5, 5, 5, 5, 5])
    mode, count = nputils._find_mode(arr)
    assert mode == 5
    assert count == 5


def test__find_mode_empty():
    arr = np.array([])
    mode, count = nputils._find_mode(arr)
    assert math.isnan(mode)
    assert count == 0


def test__find_mode_multi_dim():
    arr = np.array([[1, 2, 3], [3, 4, 5], [5, 5, 5]])
    mode, count = nputils._find_mode(arr, axis=1)
    assert mode.tolist() == [1, 3, 5]
    assert count.tolist() == [1, 1, 3]


def test__find_mode_keep_dim():
    arr = np.array([[1, 2, 3], [3, 4, 5], [5, 5, 5]])
    mode, count = nputils._find_mode(arr, axis=1, keepdims=True)
    assert mode.tolist() == [[1], [3], [5]]
    assert count.tolist() == [[1], [1], [3]]


def test_log_tensor_stats_basic(device, lsys, caplog):
    shape = [1, 6]
    src = sfnp.device_array(device, shape, dtype=sfnp.float32)
    data = [1, 2, 3, 3, 4, 5]
    src.view(0).items = data

    # Ensure array stats are logged properly
    log_messages = []
    with patch.object(
        nputils.logger,
        "debug",
        side_effect=lambda message: log_messages.append(message),
    ):
        nputils.debug_log_tensor_stats(src)
        assert log_messages[0] == "NaN count: 0 / 6"
        assert log_messages[1] == "Shape: (1, 6), dtype: float32"
        assert log_messages[2] == "Min (excluding NaN): 1.0"
        assert log_messages[3] == "Max (excluding NaN): 5.0"
        assert log_messages[4] == "Mean (excluding NaN): 3.0"
        assert log_messages[5] == "Mode (excluding NaN): 3.0"
        assert log_messages[6] == "First 10 elements: [1. 2. 3. 3. 4. 5.]"
        assert log_messages[7] == "Last 10 elements: [1. 2. 3. 3. 4. 5.]"


def test_log_tensor_stats_with_nan(device, lsys, caplog):
    shape = [1, 8]
    src = sfnp.device_array(device, shape, dtype=sfnp.float64)
    data = [3, np.nan, 4, 3, 1, np.nan, 5, 9]
    src.view(0).items = data

    # Ensure array stats are logged properly
    log_messages = []
    with patch.object(
        nputils.logger,
        "debug",
        side_effect=lambda message: log_messages.append(message),
    ):
        nputils.debug_log_tensor_stats(src)
        assert log_messages[0] == "NaN count: 2 / 8"
        assert log_messages[1] == "Shape: (1, 8), dtype: float64"
        assert log_messages[2] == "Min (excluding NaN): 1.0"
        assert log_messages[3] == "Max (excluding NaN): 9.0"
        assert log_messages[4] == "Mean (excluding NaN): 4.166666666666667"
        assert log_messages[5] == "Mode (excluding NaN): 3.0"
        assert log_messages[6] == "First 10 elements: [3. 4. 3. 1. 5. 9.]"
        assert log_messages[7] == "Last 10 elements: [3. 4. 3. 1. 5. 9.]"


def test_log_tensor_stats_empty(device, lsys, caplog):
    shape = [1, 0]
    src = sfnp.device_array(device, shape, dtype=sfnp.float32)

    # Ensure array stats are logged properly
    log_messages = []
    with patch.object(
        nputils.logger,
        "debug",
        side_effect=lambda message: log_messages.append(message),
    ):
        nputils.debug_log_tensor_stats(src)
        assert log_messages[0] == "NaN count: 0 / 0"
        assert log_messages[1] == "Shape: (1, 0), dtype: float32"


def test_log_tensor_stats_multi_dim(device, lsys, caplog):
    shape = [3, 3]
    src = sfnp.device_array(device, shape, dtype=sfnp.float32)
    data = [[1, np.nan, 3], [np.nan, 4, 5], [5, np.nan, 7]]
    for i in range(3):
        src.view(i).items = data[i]

    # Ensure array stats are logged properly
    log_messages = []
    with patch.object(
        nputils.logger,
        "debug",
        side_effect=lambda message: log_messages.append(message),
    ):
        nputils.debug_log_tensor_stats(src)
        assert log_messages[0] == "NaN count: 3 / 9"
        assert log_messages[1] == "Shape: (3, 3), dtype: float32"
        assert log_messages[2] == "Min (excluding NaN): 1.0"
        assert log_messages[3] == "Max (excluding NaN): 7.0"
        assert log_messages[4] == "Mean (excluding NaN): 4.166666507720947"
        assert log_messages[5] == "Mode (excluding NaN): 5.0"
        assert log_messages[6] == "First 10 elements: [1. 3. 4. 5. 5. 7.]"
        assert log_messages[7] == "Last 10 elements: [1. 3. 4. 5. 5. 7.]"
