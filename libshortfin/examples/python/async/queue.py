# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio

import shortfin as sf

lsys = sf.host.CPUSystemBuilder().create_system()

received_payloads = []


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
        while (counter := counter + 1) <= 500:
            msg = Message(f"Msg#{counter:03}")
            await self.writer(msg)
            print(f"Wrote message: {counter}")
        self.writer.close()


class ReaderProcess(sf.Process):
    def __init__(self, queue, **kwargs):
        super().__init__(**kwargs)
        self.reader = queue.reader()

    async def run(self):
        count = 0
        while message := await self.reader():
            print(f"[pid={self.pid}] Received message:", message)
            received_payloads.append(message.payload)
            count += 1
            # After 100 messages, let the writer get ahead of the readers.
            # Ensures that backlog and async close with a backlog works.
            if count == 100:
                await asyncio.sleep(0.25)


async def main():
    queue = sf.Queue()
    main_scope = lsys.create_scope()
    # TODO: Also test named queues.
    # queue = lsys.create_queue("infeed")
    w1 = lsys.create_worker("w1")
    w1_scope = lsys.create_scope(w1)
    await asyncio.gather(
        WriterProcess(queue, scope=main_scope).launch(),
        # By having a reader on the main worker and a separate worker,
        # we test both intra and inter worker future resolution, which
        # take different paths internally.
        ReaderProcess(queue, scope=main_scope).launch(),
        ReaderProcess(queue, scope=w1_scope).launch(),
    )


lsys.run(main())


# Validate.
# May have come in slightly out of order so sort.
received_payloads.sort()
expected_payloads = [f"Msg#{i:03}" for i in range(1, 501)]
expected_payloads.sort()

assert (
    received_payloads == expected_payloads
), f"EXPECTED: {repr(expected_payloads)}\nACTUAL:{received_payloads}"
