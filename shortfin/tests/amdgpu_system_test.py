# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import shortfin as sf


@pytest.mark.system("amdgpu")
def test_create_amd_gpu_system_defaults():
    sc = sf.amdgpu.SystemBuilder()
    with sc.create_system() as ls:
        print(f"DEFAULTS:", ls)
        for device_name in ls.device_names:
            print(f"  DEVICE: {device_name} = {ls.device(device_name)}")
        assert "amdgpu:0:0@0" in ls.device_names
        assert "hostcpu:0:0@0" not in ls.device_names


@pytest.mark.system("amdgpu")
def test_create_amd_gpu_tracing_level():
    sc = sf.amdgpu.SystemBuilder()
    assert sc.tracing_level == 2  # Default
    sc = sf.amdgpu.SystemBuilder(amdgpu_tracing_level=1)
    assert sc.tracing_level == 1


@pytest.mark.system("amdgpu")
def test_create_amd_gpu_allocator():
    sc = sf.amdgpu.SystemBuilder(allocators="caching;debug")
    assert sc.amdgpu_allocator_specs == ["caching", "debug"]
    with sc.create_system() as ls:
        # Nothing to verify
        pass


@pytest.mark.system("amdgpu")
def test_create_amd_gpu_logical_devices_per_physical_device():
    # Default.
    sc = sf.amdgpu.SystemBuilder()
    assert sc.logical_devices_per_physical_device == 1

    # Override.
    sc = sf.amdgpu.SystemBuilder(amgdpu_logical_devices_per_physical_device=2)
    assert sc.logical_devices_per_physical_device == 2
    sc.visible_devices = sc.available_devices[0:1]
    with sc.create_system() as ls:
        assert "amdgpu:0:0@0" in ls.device_names
        assert "amdgpu:0:0@1" in ls.device_names


@pytest.mark.system("amdgpu")
def test_create_amd_gpu_system_defaults():
    sc = sf.amdgpu.SystemBuilder(amdgpu_cpu_devices_enabled=True)
    with sc.create_system() as ls:
        print(f"WITH CPU:", ls)
        for device_name in ls.device_names:
            print(f"  DEVICE: {device_name} = {ls.device(device_name)}")
        assert "amdgpu:0:0@0" in ls.device_names
        assert "hostcpu:0:0@0" in ls.device_names


@pytest.mark.system("amdgpu")
def test_create_amd_gpu_system_visible():
    sc_query = sf.amdgpu.SystemBuilder()
    available = sc_query.available_devices
    print("AVAILABLE:", available)

    # Create a system with the explicitly listed available device.
    sc_query.visible_devices = [available[0]]
    with sc_query.create_system() as ls:
        assert "amdgpu:0:0@0" in ls.device_names
        assert len(ls.devices) == 1

    # Create via option.
    sc = sf.amdgpu.SystemBuilder(amdgpu_visible_devices=available[0])
    with sc.create_system() as ls:
        assert "amdgpu:0:0@0" in ls.device_names
        assert len(ls.devices) == 1

    # Duplicates not available.
    sc = sf.amdgpu.SystemBuilder(
        amdgpu_visible_devices=";".join(available[0] for i in range(100))
    )
    with pytest.raises(
        ValueError, match="was requested more times than present on the system"
    ):
        sc.create_system()


@pytest.mark.system("amdgpu")
def test_create_amd_gpu_system_visible_unknown():
    sc = sf.amdgpu.SystemBuilder(amdgpu_visible_devices="foobar")
    with pytest.raises(
        ValueError,
        match="Requested visible device 'foobar' was not found on the system",
    ):
        sc.create_system()


@pytest.mark.system("amdgpu")
def test_system_ctor():
    with sf.System("amdgpu") as ls:
        assert "amdgpu:0:0@0" in ls.device_names
