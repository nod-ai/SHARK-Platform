#!/usr/bin/env python
# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio

import shortfin as sf

lsys = sf.host.CPUSystemBuilder().create_system()


class MyProcess(sf.Process):
    def __init__(self, scope, arg):
        super().__init__(scope)
        self.arg = arg

    async def run(self):
        print("Hello async:", self.arg, self)
        processes = []
        if self.arg < 10:
            await asyncio.sleep(0.3)
            processes.append(MyProcess(self.scope, self.arg + 1).launch())
        await asyncio.gather(*processes)


async def main():
    worker = lsys.create_worker("main")
    scope = lsys.create_scope(worker)
    processes = []
    for i in range(10):
        processes.append(MyProcess(scope, i).launch())
        await asyncio.sleep(0.1)
        processes.append(MyProcess(scope, i * 100).launch())
        await asyncio.sleep(0.1)
        processes.append(MyProcess(scope, i * 1000).launch())

    print("<<MAIN WAITING>>")
    await asyncio.gather(*processes)
    print("** MAIN DONE **")
    return i


print("RESULT:", lsys.run(main()))
