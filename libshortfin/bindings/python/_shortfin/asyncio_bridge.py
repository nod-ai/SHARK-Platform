# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio

from . import lib as sfl


class PyWorkerEventLoop(asyncio.AbstractEventLoop):
    def __init__(self, worker: sfl.local.Worker):
        self._worker = worker

    def get_debug(self):
        # Requirement of asyncio.
        return False

    def create_task(self, coro):
        return asyncio.Task(coro, loop=self)

    def call_soon_threadsafe(self, callback, *args, context=None) -> asyncio.Handle:
        def on_worker():
            asyncio.set_event_loop(self)
            return callback(*args)

        self._worker.call_threadsafe(on_worker)
        # TODO: Return future.

    def call_soon(self, callback, *args, context=None) -> asyncio.Handle:
        if not args:
            self._worker.call(callback)
        else:

            def trampoline():
                callback(*args)

            self._worker.call(trampoline)

    def call_exception_handler(self, context) -> None:
        # TODO: Should route this to the central exception handler.
        raise RuntimeError(f"Async exception on {self._worker}: {context}")
