# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


@pytest.mark.requires_amd_gpu
def test_create_amd_gpu_system():
    from _shortfin import lib as sfl

    sc = sfl.local.amdgpu.SystemBuilder()
    ls = sc.create_system()
    print(f"LOCAL SYSTEM:", ls)
    for device_name in ls.device_names:
        print(f"  DEVICE: {device_name} = {ls.device(device_name)}")

    print(ls.devices)
    print("Shutting down")
    ls.shutdown()
