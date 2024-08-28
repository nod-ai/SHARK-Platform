#!/usr/bin/env python
# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
from pathlib import Path
import sys

import shortfin as sf
import shortfin.array as sfnp

MAX_BATCH = 1


class InferenceRequest(sf.Message):
    def __init__(self, raw_image_data):
        super().__init__()
        self.raw_image_data = raw_image_data


class InferenceProcess(sf.Process):
    def __init__(self, program, request_queue, **kwargs):
        super().__init__(**kwargs)
        self.program = program
        self.request_reader = request_queue.reader()
        self.device = self.scope.device(0)
        self.device_input = sfnp.device_array(
            self.device, [MAX_BATCH, 3, 224, 224], sfnp.float32
        )
        self.host_staging = self.device_input.for_transfer()

    async def run(self):
        print(f"Inference process: {self.pid}")
        while request := await self.request_reader():
            print(f"[{self.pid}] Got request {request}")
            # TODO: Should really be taking a slice and writing that. For now,
            # just writing to the backing storage is the best we have API
            # support for. Generally, APIs on storage should be mirrored onto
            # the array.
            self.host_staging.storage.data = request.raw_image_data
            print(self.host_staging)
            self.device_input.storage.copy_from(self.host_staging.storage)
            print(self.device_input)


class Main:
    def __init__(self, lsys: sf.System, home_dir: Path):
        self.processes_per_worker = 1
        self.lsys = lsys
        self.home_dir = home_dir
        self.request_queue = lsys.create_queue("request")
        self.program_module = self.lsys.load_module(home_dir / "model.vmfb")
        print(f"Loaded: {self.program_module}")
        self.processes = []

    async def start_scope(self, scope):
        # Note that currently, program load is synchronous. But we do it
        # in a task so we can await it in the future and let program loads
        # overlap.
        program = scope.load_unbound_program([self.program_module])
        for _ in range(self.processes_per_worker):
            self.processes.append(
                InferenceProcess(program, self.request_queue, scope=scope).launch()
            )

    async def main(self):
        devices = self.lsys.devices
        print(
            f"System created with {len(devices)} devices:\n  "
            f"{'  '.join(repr(d) for d in devices)}"
        )
        # We create a physical worker and initial scope for each device.
        # This isn't a hard requirement and there are advantages to other
        # topologies.
        initializers = []
        for device in devices:
            worker = self.lsys.create_worker(f"device-{device.name}")
            scope = self.lsys.create_scope(worker, devices=[device])
            initializers.append(self.start_scope(scope))

        # Run all initializers in parallel. These launch inference processes.
        print("Waiting for initializers")
        await asyncio.gather(*initializers)

        # Wait for inference processes to end.
        print(f"Running {len(self.processes)} inference processes")
        await asyncio.gather(*self.processes)
        print("Inference processors completed")


def run_cli(home_dir: Path, argv):
    def client():
        # Create a random image.
        print("Preparing requests...")
        writer = main.request_queue.writer()

        # Dumb way to prepare some data to feed [1, 3, 224, 224] f32.
        import array

        dummy_data = array.array(
            "f", ([0.2] * (224 * 224)) + ([0.4] * (224 * 224)) + ([-0.2] * (224 * 224))
        )
        # dummy_data = array.array("f", [0.2] * (3 * 224 * 224))
        message = InferenceRequest(dummy_data)
        writer(message)

        # Done.
        writer.close()

    lsys = sf.host.CPUSystemBuilder().create_system()
    main = Main(lsys, home_dir)
    lsys.init_worker.call_threadsafe(client)
    lsys.run(main.main())


if __name__ == "__main__":
    home_dir = Path(__file__).resolve().parent
    run_cli(home_dir, sys.argv[1:])
