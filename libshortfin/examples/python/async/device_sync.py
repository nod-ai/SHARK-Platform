#!/usr/bin/env python
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import array
import asyncio

import shortfin as sf
import shortfin.array as snp


class MyProcess(sf.Process):
    async def run(self):
        device = self.fiber.device(0)
        ary1 = snp.device_array(device, [32, 1, 4], snp.int32)
        ary1.storage.fill(array.array("i", [0]))
        print(f"[pid:{self.pid}] ARY1:", ary1)
        await device
        print(f"[pid:{self.pid}] Device sync fill0")
        ary1.storage.fill(array.array("i", [1]))
        await device
        print(f"[pid:{self.pid}] Device sync fill1")


async def main():
    worker = lsys.create_worker("main")
    fiber = lsys.create_fiber(worker)
    print("+++ Launching process")
    await asyncio.gather(
        MyProcess(fiber=fiber).launch(),
        MyProcess(fiber=fiber).launch(),
    )
    print("--- Process terminated")


lsys = sf.host.CPUSystemBuilder().create_system()
# lsys = sf.amdgpu.SystemBuilder().create_system()
lsys.run(main())
