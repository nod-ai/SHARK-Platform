#!/usr/bin/env python
# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import threading
import time

import shortfin as sf

lsys = sf.host.CPUSystemBuilder().create_system()
worker = lsys.create_worker("main")
scope = lsys.create_scope(worker)
print("Worker:", worker)

p = sf.Process(scope)
print(p)

ps = []


class MyProcess(sf.Process):
    def __init__(self, scope, arg):
        super().__init__(scope)
        self.arg = arg

    async def run(self):
        print("Hello async:", self.arg, self)
        if self.arg < 10:
            await asyncio.sleep(0.2)
            MyProcess(self.scope, self.arg + 1).launch()


p2 = MyProcess(scope, 0)
print(p2, p2.__class__)
p2.launch()

import time

time.sleep(5)
