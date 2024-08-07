# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from _shortfin import lib as sfl


@pytest.fixture
def lsys():
    sc = sfl.host.CPUSystemBuilder()
    return sc.create_local_system()


@pytest.fixture
def scope(lsys):
    return lsys.create_scope()


def test_device_access(scope):
    first_name = scope.device_names[0]
    assert first_name == "cpu0"
    first_device = scope.device(0)  # By index
    assert isinstance(first_device, sfl.host.HostCPUDevice)
    assert first_device is scope.device(first_name)  # By name
    print(first_device)
    devices = scope.devices
    named_devices = scope.named_devices
    assert first_name in named_devices
    assert devices[0] is named_devices[first_name]
    assert devices[0] is first_device
    with pytest.raises(ValueError):
        scope.device("cpu1")
    with pytest.raises(ValueError):
        scope.device(1)

    # Access via devices pseudo collection.
    assert scope.devices.cpu0 is first_device
    assert scope.devices[0] is first_device
    assert scope.devices["cpu0"] is first_device
    assert len(scope.devices) == 1
    with pytest.raises(ValueError):
        scope.devices.cpu1
    with pytest.raises(ValueError):
        scope.devices[1]


def test_device_affinity_repr(scope):
    assert (
        repr(sfl.DeviceAffinity(scope.device(0)))
        == "DeviceAffinity(host-cpu:0:0@0[0x1])"
    )
    assert repr(sfl.DeviceAffinity()) == "DeviceAffinity(ANY)"


def test_device_affinity_resolve(scope):
    # TODO: Need a scope with multiple devices to test errors.
    print(scope.device_affinity(0, "cpu0", scope.device(0)))
