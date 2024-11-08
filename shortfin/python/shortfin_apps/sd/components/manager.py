# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import threading

import shortfin as sf

logger = logging.getLogger(__name__)


def get_selected_devices(sb: sf.SystemBuilder, device_ids=None):
    available = sb.available_devices
    selected = []
    if device_ids is not None:
        if len(device_ids) >= len(available):
            raise ValueError(
                f"Requested more device ids ({device_ids}) than available ({available})."
            )
        for did in device_ids:
            if isinstance(did, str):
                try:
                    did = int(did)
                except ValueError:
                    did = did
            if did in available:
                selected.append(did)
            elif isinstance(did, int):
                selected.append(available[did])
            else:
                raise ValueError(f"Device id {did} could not be parsed.")
    else:
        selected = available
    return selected


class SystemManager:
    def __init__(self, device="local-task", device_ids=None, async_allocs=True):
        if any(x in device for x in ["local-task", "cpu"]):
            self.ls = sf.host.CPUSystemBuilder().create_system()
        elif any(x in device for x in ["hip", "amdgpu"]):
            sb = sf.SystemBuilder(
                system_type="amdgpu", amdgpu_async_allocations=async_allocs
            )
            if device_ids:
                sb.visible_devices = sb.available_devices
                sb.visible_devices = get_selected_devices(sb, device_ids)
            self.ls = sb.create_system()
        logger.info(f"Created local system with {self.ls.device_names} devices")
        # TODO: Come up with an easier bootstrap thing than manually
        # running a thread.
        self.t = threading.Thread(target=lambda: self.ls.run(self.run()))
        self.command_queue = self.ls.create_queue("command")
        self.command_writer = self.command_queue.writer()

    def start(self):
        logger.info("Starting system manager")
        self.t.start()

    def shutdown(self):
        logger.info("Shutting down system manager")
        self.command_queue.close()

    async def run(self):
        reader = self.command_queue.reader()
        while command := await reader():
            ...
        logging.info("System manager command processor stopped")
