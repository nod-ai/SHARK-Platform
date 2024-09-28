# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import time

from _shortfin import lib as sfl


@pytest.fixture
def lsys():
    sc = sfl.local.host.CPUSystemBuilder()
    ls = sc.create_system()
    yield ls
    ls.shutdown()


@pytest.fixture
def fiber(lsys):
    # TODO: Should adopt the main thread.
    worker = lsys.create_worker("main")
    return lsys.create_fiber(worker)


def test_raw_device_access(fiber):
    first_name = fiber.device_names[0]
    assert first_name == "cpu0"
    first_device = fiber.raw_device(0)  # By index
    assert isinstance(first_device, sfl.local.host.HostCPUDevice)
    assert first_device is fiber.raw_device(first_name)  # By name
    print(first_device)
    named_devices = fiber.devices_dict
    assert first_name in named_devices
    with pytest.raises(ValueError):
        fiber.raw_device("cpu1")
    with pytest.raises(ValueError):
        fiber.raw_device(1)


def test_devices_collection_access(fiber):
    # # Access via devices pseudo collection.
    first_device = fiber.raw_device(0)
    assert fiber.devices.cpu0.raw_device is first_device
    assert fiber.devices[0].raw_device is first_device
    assert fiber.devices["cpu0"].raw_device is first_device
    assert len(fiber.devices) == 1
    with pytest.raises(ValueError):
        fiber.devices.cpu1
    with pytest.raises(ValueError):
        fiber.devices[1]
    iter_list = list(fiber.devices)
    assert iter_list == [fiber.device(0)]


def test_device_affinity_repr(fiber):
    assert (
        repr(sfl.local.DeviceAffinity(fiber.raw_device(0)))
        == "DeviceAffinity(hostcpu:0:0@0[0x1])"
    )
    assert repr(sfl.local.DeviceAffinity()) == "DeviceAffinity(ANY)"


def test_device_affinity_resolve(fiber):
    # TODO: Need a fiber with multiple devices to test errors.
    print(fiber.device(0, "cpu0", fiber.raw_device(0)))
