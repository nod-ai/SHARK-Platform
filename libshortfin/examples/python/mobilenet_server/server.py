#!/usr/bin/env python
# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
from pathlib import Path

import shortfin as sf


class InferenceProcess(sf.Process):
    def __init__(self, program, **kwargs):
        super().__init__(**kwargs)
        self.program = program

    async def run(self):
        print(f"Inference process: {self.pid}")


class Main:
    def __init__(self, lsys: sf.System, home_dir: Path):
        self.processes_per_worker = 4
        self.lsys = lsys
        self.home_dir = home_dir
        self.program_module = self.lsys.load_module(home_dir / "model.vmfb")
        print(f"Loaded: {self.program_module}")
        self.processes = []

    async def initialize(self, scope):
        # Note that currently, program load is synchronous. But we do it
        # in a task so we can await it in the future and let program loads
        # overlap.
        program = scope.load_unbound_program([self.program_module])
        for _ in range(self.processes_per_worker):
            self.processes.append(InferenceProcess(program, scope=scope).launch())

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
            initializers.append(self.initialize(scope))

        # Run all initializers in parallel. These launch inference processes.
        await asyncio.gather(*initializers)

        # Wait for inference processes to end.
        await asyncio.gather(*self.processes)


def run_server(home_dir: Path):
    lsys = sf.host.CPUSystemBuilder().create_system()
    main = Main(lsys, home_dir)
    lsys.run(main.main())


if __name__ == "__main__":
    home_dir = Path(__file__).resolve().parent
    run_server(home_dir)
