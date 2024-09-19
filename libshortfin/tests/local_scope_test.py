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
def scope(lsys):
    # TODO: Should adopt the main thread.
    worker = lsys.create_worker("main")
    return lsys.create_scope(worker)


def test_raw_device_access(scope):
    first_name = scope.device_names[0]
    assert first_name == "cpu0"
    first_device = scope.raw_device(0)  # By index
    assert isinstance(first_device, sfl.local.host.HostCPUDevice)
    assert first_device is scope.raw_device(first_name)  # By name
    print(first_device)
    named_devices = scope.devices_dict
    assert first_name in named_devices
    with pytest.raises(ValueError):
        scope.raw_device("cpu1")
    with pytest.raises(ValueError):
        scope.raw_device(1)


def test_devices_collection_access(scope):
    # # Access via devices pseudo collection.
    first_device = scope.raw_device(0)
    assert scope.devices.cpu0.raw_device is first_device
    assert scope.devices[0].raw_device is first_device
    assert scope.devices["cpu0"].raw_device is first_device
    assert len(scope.devices) == 1
    with pytest.raises(ValueError):
        scope.devices.cpu1
    with pytest.raises(ValueError):
        scope.devices[1]
    iter_list = list(scope.devices)
    assert iter_list == [scope.device(0)]


def test_device_affinity_repr(scope):
    assert (
        repr(sfl.local.DeviceAffinity(scope.raw_device(0)))
        == "DeviceAffinity(host-cpu:0:0@0[0x1])"
    )
    assert repr(sfl.local.DeviceAffinity()) == "DeviceAffinity(ANY)"


def test_device_affinity_resolve(scope):
    # TODO: Need a scope with multiple devices to test errors.
    print(scope.device(0, "cpu0", scope.raw_device(0)))
