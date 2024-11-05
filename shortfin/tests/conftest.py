# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import pytest
import shlex

import shortfin as sf


def pytest_addoption(parser):
    parser.addoption(
        "--system",
        action="store",
        metavar="NAME",
        default="hostcpu",
        help="Enable tests for system name ('hostcpu', 'amdgpu', ...)",
    )
    parser.addoption(
        "--compile-flags",
        action="store",
        metavar="FLAGS",
        help="Compile flags to run test on the --system (required if it cannot be inferred)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "system(name): mark test to run only on a named system"
    )
    config.addinivalue_line(
        "markers", "slow: mark test to run in a separate, slow suite."
    )


def pytest_runtest_setup(item):
    system_type = item.config.getoption("--system")
    # Filter tests based on system mark.
    required_system_names = [mark.args[0] for mark in item.iter_markers("system")]
    if required_system_names:
        if not all(name == system_type for name in required_system_names):
            pytest.skip(
                f"test requires system in {required_system_names!r} but has "
                f"{system_type!r} (set with --system arg)"
            )
    # Set the default.
    sf.SystemBuilder.default_system_type = system_type


# Keys that will be cleaned project wide prior to and after each test run.
# Test code can freely modify these.
CLEAN_ENV_KEYS = [
    "SHORTFIN_HOSTCPU_TOPOLOGY_NODES",
    "SHORTFIN_HOSTCPU_TOPOLOGY_MAX_GROUP_COUNT",
    "SHORTFIN_SYSTEM_TYPE",
]


@pytest.fixture(scope="session")
def compile_flags(pytestconfig) -> list[str]:
    compile_flags = pytestconfig.getoption("--compile-flags")
    if compile_flags is not None:
        return shlex.split(compile_flags)
    # Try to figure it out from the system.
    system_type = pytestconfig.getoption("--system")
    if system_type == "hostcpu":
        return [
            "--iree-hal-target-device=llvm-cpu",
            "--iree-llvmcpu-target-cpu=host",
        ]
    pytest.skip(
        reason="Test needs to compile a binary and no --compile-flags set (or "
        "could not be inferred)"
    )


@pytest.fixture(autouse=True)
def clean_env():
    def kill():
        for key in CLEAN_ENV_KEYS:
            if key in os.environ:
                del os.environ[key]
                os.unsetenv(key)

    kill()
    yield
    kill()


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
