#!/usr/bin/env python
# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import threading
import time

from _shortfin import asyncio_bridge
from _shortfin import lib as sfl

lsys = sfl.local.host.CPUSystemBuilder().create_system()
worker = lsys.create_worker("main")
print("Worker:", worker)


@worker.call
def print_thread_id():
    print(f"Worker (tid={threading.get_ident()})")


async def do_something():
    print(f"FROM ASYNC do_something (tid={threading.get_ident()})")
    raise ValueError("Yeah, we're going to need you to come in on Saturday")


f = asyncio.run_coroutine_threadsafe(do_something(), worker.loop)
print("FUTURE:", f)


print(f"Sleeping... (tid={threading.get_ident()})")
time.sleep(2)
print("FUTURE NOW:", f.result())
