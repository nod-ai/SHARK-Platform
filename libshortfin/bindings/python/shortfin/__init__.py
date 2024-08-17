# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from _shortfin import lib as _sfl

# Most classes from the native "local" namespace are aliased to the top
# level of the public API.
Device = _sfl.local.Device
Node = _sfl.local.Node
Process = _sfl.local.Process
Scope = _sfl.local.Scope
ScopedDevice = _sfl.local.ScopedDevice
SingleWaitFuture = _sfl.local.SingleWaitFuture
System = _sfl.local.System
SystemBuilder = _sfl.local.SystemBuilder
Worker = _sfl.local.Worker

# Array is auto-imported.
from . import array

# System namespaces.
from . import amdgpu
from . import host

__all__ = [
    "Device",
    "Node",
    "Scope",
    "ScopedDevice",
    "SingleWaitFuture",
    "System",
    "SystemBuilder",
    "Worker",
    # System namespaces.
    "amdgpu",
    "host",
]
