# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio

import shortfin as sf

lsys = sf.host.CPUSystemBuilder().create_system()


class Message(sf.Message):
    def __init__(self, payload):
        super().__init__()
        self.payload = payload

    def __repr__(self):
        return f"Message(payload='{self.payload}')"


class WriterProcess(sf.Process):
    def __init__(self, queue, **kwargs):
        super().__init__(**kwargs)
        self.writer = queue.writer()

    async def run(self):
        print("Start writer")
        counter = 0
        while True:
            await asyncio.sleep(0.01)
            counter += 1
            msg = Message(f"Msg#{counter}")
            await self.writer(msg)
            print(f"Wrote message: {counter}")


class ReaderProcess(sf.Process):
    def __init__(self, queue, **kwargs):
        super().__init__(**kwargs)
        self.reader = queue.reader()

    async def run(self):
        while True:
            message = await self.reader()
            print("Received message:", message)


async def main():
    queue = lsys.create_queue("infeed")
    scope = lsys.create_scope()
    await asyncio.gather(
        WriterProcess(queue, scope=scope).launch(),
        ReaderProcess(queue, scope=scope).launch(),
    )


lsys.run(main())
