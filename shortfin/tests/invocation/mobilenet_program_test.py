# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import array
import asyncio
import time
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
def fiber0(lsys):
    return lsys.create_fiber()


@pytest.fixture
def device(fiber0):
    return fiber0.device(0)


@pytest.fixture
def mobilenet_program_function(
    lsys, mobilenet_compiled_cpu_path
) -> tuple[sf.ProgramFunction]:
    program_module = lsys.load_module(mobilenet_compiled_cpu_path)
    program = sf.Program([program_module], devices=lsys.devices)
    main_function = program["module.torch-jit-export"]
    return main_function


def get_mobilenet_ref_input(device) -> sfnp.device_array:
    dummy_data = array.array(
        "f", ([0.2] * (224 * 224)) + ([0.4] * (224 * 224)) + ([-0.2] * (224 * 224))
    )
    device_input = sfnp.device_array(device, [1, 3, 224, 224], sfnp.float32)
    staging_input = device_input.for_transfer()
    with staging_input.map(discard=True) as m:
        m.fill(dummy_data)
    device_input.copy_from(staging_input)
    return device_input


async def assert_mobilenet_ref_output(device, device_output):
    host_output = device_output.for_transfer()
    host_output.copy_from(device_output)
    await device
    flat_output = host_output.items
    absmean = functools.reduce(
        lambda x, y: x + abs(y) / len(flat_output), flat_output, 0.0
    )
    print("RESULT:", absmean)
    assert absmean == pytest.approx(5.01964943873882)


def test_invoke_mobilenet(lsys, fiber0, mobilenet_program_function):
    device = fiber0.device(0)

    async def main():
        device_input = get_mobilenet_ref_input(device)
        (device_output,) = await mobilenet_program_function(device_input, fiber=fiber0)
        await assert_mobilenet_ref_output(device, device_output)

    lsys.run(main())


def test_invoke_mobilenet_multi_fiber(lsys, mobilenet_program_function):
    class InferProcess(sf.Process):
        async def run(self):
            start_time = time.time()

            def duration():
                return round((time.time() - start_time) * 1000.0)

            print(f"{self}: Start")
            device = self.fiber.device(0)
            device_input = get_mobilenet_ref_input(device)
            (device_output,) = await mobilenet_program_function(
                device_input, fiber=self.fiber
            )
            print(f"{self}: Program complete (+{duration()}ms)")
            await assert_mobilenet_ref_output(device, device_output)
            print(f"{self} End (+{duration()}ms)")

    async def main():
        start_time = time.time()

        def duration():
            return round((time.time() - start_time) * 1000.0)

        fibers = [lsys.create_fiber() for _ in range(5)]
        print("Fibers:", fibers)
        processes = [InferProcess(fiber=f).launch() for f in fibers]
        print("Waiting for processes:", processes)
        await asyncio.gather(*processes)
        print(f"All processes complete: (+{duration()}ms)")

    lsys.run(main())
