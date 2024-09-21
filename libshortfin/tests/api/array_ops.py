# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import array
import math
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


def test_argmax(device):
    src = sfnp.device_array(device, [4, 16, 128], dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod([1, 16, 128]))]
    for i in range(4):
        src.view(i).items = data
        data.reverse()

    # default variant
    result = sfnp.argmax(src)
    assert result.shape == [4, 16]
    assert result.view(0).items.tolist() == [127] * 16
    assert result.view(1).items.tolist() == [0] * 16
    assert result.view(2).items.tolist() == [127] * 16
    assert result.view(3).items.tolist() == [0] * 16

    # keepdims variant
    result = sfnp.argmax(src, keepdims=True)
    assert result.shape == [4, 16, 1]

    # out= variant
    out = sfnp.device_array(device, [4, 16], dtype=sfnp.int64)
    sfnp.argmax(src, out=out)
    assert out.shape == [4, 16]
    assert out.view(0).items.tolist() == [127] * 16
    assert out.view(1).items.tolist() == [0] * 16
    assert out.view(2).items.tolist() == [127] * 16
    assert out.view(3).items.tolist() == [0] * 16

    # out= keepdims variant
    out = sfnp.device_array(device, [4, 16, 1], dtype=sfnp.int64)
    sfnp.argmax(src, keepdims=True, out=out)
    assert out.shape == [4, 16, 1]
    assert out.view(0).items.tolist() == [127] * 16
    assert out.view(1).items.tolist() == [0] * 16
    assert out.view(2).items.tolist() == [127] * 16
    assert out.view(3).items.tolist() == [0] * 16


@pytest.mark.parametrize(
    "dtype",
    [
        sfnp.float16,
        sfnp.float32,
    ],
)
def test_argmax_dtypes(device, dtype):
    # Just verifies that the dtype functions. We don't have IO support for
    # some of these.
    src = sfnp.device_array(device, [4, 16, 128], dtype=dtype)
    sfnp.argmax(src)
