# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import pytest

import shortfin as sf

@pytest.fixture(autouse=True)
def clean_env():
    save = {}
    def kill():
        for key, value in os.environ.items():
            if key.startswith("SHORTFIN_"):
                save[key] = value
                del os.environ[key]
    kill()
    yield
    kill()
    for key, value in save.items():
        os.environ[key] = value


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


def test_create_host_cpu_system_env_vars():
    os.putenv("SHORTFIN_HOSTCPU_TOPOLOGY_NODES", "0,0")
    os.putenv("SHORTFIN_HOSTCPU_TOPOLOGY_MAX_GROUP_COUNT", "2")
    sc = sf.host.CPUSystemBuilder()
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
