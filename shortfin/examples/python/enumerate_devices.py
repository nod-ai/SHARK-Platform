# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

r"""Simple test program that enumerates devices available.

Run with SystemBuilder keyword args on the command line like::

  python examples/python/enumerate_devices.py \
    system_type=amdgpu amdgpu_logical_devices_per_physical_device=4

"""

import sys

import shortfin as sf


def main():
    args = [arg.split("=", maxsplit=1) for arg in sys.argv[1:]]
    arg_dict = {k: v for k, v in args}
    print(f"Creating system with args: {arg_dict}")
    builder = sf.SystemBuilder(**arg_dict)
    with builder.create_system() as ls:
        for device in ls.devices:
            print(device)


if __name__ == "__main__":
    main()
