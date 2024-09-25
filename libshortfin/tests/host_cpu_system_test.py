# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import shortfin as sf


def test_create_host_cpu_system_defaults():
    sc = sf.host.CPUSystemBuilder()
    with sc.create_system() as ls:
        print(f"LOCAL SYSTEM:", ls)
        print("\n".join(repr(d) for d in ls.devices))
        assert len(ls.devices) > 0


def test_create_host_cpu_system_topology_nodes_all():
    sc = sf.host.CPUSystemBuilder(
        hostcpu_topology_nodes="all", hostcpu_topology_max_group_count=2
    )
    with sc.create_system() as ls:
        print(f"LOCAL SYSTEM:", ls)
        print("\n".join(repr(d) for d in ls.devices))
        assert len(ls.devices) > 0


def test_create_host_cpu_system_topology_nodes_explicit():
    sc = sf.host.CPUSystemBuilder(
        hostcpu_topology_nodes="0,0", hostcpu_topology_max_group_count=2
    )
    with sc.create_system() as ls:
        print(f"LOCAL SYSTEM:", ls)
        print("\n".join(repr(d) for d in ls.devices))
        assert len(ls.devices) == 2


def test_create_host_cpu_system_unsupported_option():
    sc = sf.host.CPUSystemBuilder(unsupported="foobar")
    with pytest.raises(
        ValueError, match="Specified options were not used: unsupported"
    ):
        sc.create_system()
