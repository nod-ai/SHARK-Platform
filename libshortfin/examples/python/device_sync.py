#!/usr/bin/env python
# Copyright 2024 Advanced Micro Devices, Inc
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
        device = self.scope.device(0)
        ary1 = snp.device_array(device, [32, 1, 4], snp.int32)
        ary1.storage.fill(array.array("i", [0]))
        print("ARY1:", ary1)
        await device


async def main():
    worker = lsys.create_worker("main")
    scope = lsys.create_scope(worker)
    print("+++ Launching process")
    await MyProcess(scope).launch()
    print("--- Process terminated")


# lsys = sf.host.CPUSystemBuilder().create_system()
lsys = sf.amdgpu.SystemBuilder().create_system()
lsys.run(main())
