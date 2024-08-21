# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import threading

import shortfin as sf

lsys = sf.host.CPUSystemBuilder().create_system()


class MyProcess(sf.Process):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.queue = lsys.create_queue("infeed")
        print("INFEED:", self.queue)

    async def run(self):
        print("Process")


async def main():
    await MyProcess(scope=lsys.create_scope()).launch()


lsys.run(main())
