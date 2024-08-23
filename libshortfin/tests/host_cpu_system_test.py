# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


def test_create_host_cpu_system():
    from _shortfin import lib as sfl

    sc = sfl.local.host.CPUSystemBuilder()
    ls = sc.create_system()
    print(f"LOCAL SYSTEM:", ls)

    print("Sleeping in Python")
    ls.shutdown()
