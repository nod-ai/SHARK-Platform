# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import shortfin as sf


def pytest_addoption(parser):
    parser.addoption(
        "--system",
        action="store",
        metavar="NAME",
        nargs="*",
        help="Enable tests for system name ('amdgpu', ...)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "system(name): mark test to run only on a named system"
    )


def pytest_runtest_setup(item):
    required_system_names = [mark.args[0] for mark in item.iter_markers("system")]
    if required_system_names:
        available_system_names = item.config.getoption("--system") or []
        if not all(name in available_system_names for name in required_system_names):
            pytest.skip(
                f"test requires system in {required_system_names!r} but has "
                f"{available_system_names!r} (set with --system arg)"
            )


@pytest.fixture
def cpu_lsys():
    sc = sf.host.CPUSystemBuilder()
    lsys = sc.create_system()
    yield lsys
    lsys.shutdown()


@pytest.fixture
def cpu_fiber(cpu_lsys):
    return cpu_lsys.create_fiber()


@pytest.fixture
def cpu_device(cpu_fiber):
    return cpu_fiber.device(0)
