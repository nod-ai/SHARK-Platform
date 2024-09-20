# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import array
import functools
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


def test_invoke_mobilenet(lsys, scope, mobilenet_compiled_cpu_path):
    device = scope.device(0)
    dummy_data = array.array(
        "f", ([0.2] * (224 * 224)) + ([0.4] * (224 * 224)) + ([-0.2] * (224 * 224))
    )
    program_module = lsys.load_module(mobilenet_compiled_cpu_path)
    program = sf.Program([program_module], scope=scope)
    main_function = program["module.torch-jit-export"]

    async def main():
        device_input = sfnp.device_array(device, [1, 3, 224, 224], sfnp.float32)
        staging_input = device_input.for_transfer()
        with staging_input.map(discard=True) as m:
            m.fill(dummy_data)
        device_input.copy_from(staging_input)
        (device_output,) = await main_function(device_input)
        host_output = device_output.for_transfer()
        host_output.copy_from(device_output)
        await device
        flat_output = host_output.items
        absmean = functools.reduce(
            lambda x, y: x + abs(y) / len(flat_output), flat_output, 0.0
        )
        assert absmean == pytest.approx(5.01964943873882)

    lsys.run(main())
