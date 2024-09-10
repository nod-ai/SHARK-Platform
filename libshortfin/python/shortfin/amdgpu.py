# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from _shortfin import lib as _sfl

_is_available = False


def is_available():
    return _is_available


if hasattr(_sfl.local, "amdgpu"):
    AMDGPUDevice = _sfl.local.amdgpu.AMDGPUDevice
    SystemBuilder = _sfl.local.amdgpu.SystemBuilder

    __all__ = [
        "AMDGPUDevice",
        "SystemBuilder",
    ]
    _is_available = True
