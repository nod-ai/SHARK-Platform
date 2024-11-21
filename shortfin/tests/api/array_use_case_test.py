# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import array
import math
import pytest

import shortfin as sf
import shortfin.array as sfnp


@pytest.fixture
def lsys():
    # TODO: Port this test to use memory type independent access. It currently
    # presumes unified memory.
    # sc = sf.SystemBuilder()
    sc = sf.host.CPUSystemBuilder()
    lsys = sc.create_system()
    yield lsys
    lsys.shutdown()


@pytest.fixture
def fiber(lsys):
    return lsys.create_fiber()


@pytest.fixture
def device(fiber):
    return fiber.device(0)


# Tests a typical image conversion from a model oriented layout to an array
# of contained images.
def test_image_to_bytes(device):
    bs = 2
    height = 16
    width = 12
    images_shape = [bs, 3, height, width]
    images_planar = sfnp.device_array.for_host(device, images_shape, sfnp.float32)
    # Band the data so that each channel increases by 0.1 across images.
    for i in range(bs):
        for j in range(3):
            data = [i * 0.3 + j * 0.1 for _ in range(height * width)]
            images_planar.view(i, j).items = data
    images_planar = sfnp.convert(images_planar, dtype=sfnp.float16)

    # Extract and convert each image to interleaved RGB bytes.
    for idx in range(images_planar.shape[0]):
        image_planar = images_planar.view(idx)
        assert image_planar.shape == [1, 3, 16, 12]
        image_interleaved = sfnp.transpose(image_planar, (0, 2, 3, 1))
        assert image_interleaved.shape == [1, 16, 12, 3]
        image_scaled = sfnp.multiply(image_interleaved, 255)
        image = sfnp.round(image_scaled, dtype=sfnp.uint8)
        print(image)
        image_bytes = bytes(image.map(read=True))
        print(image_bytes)
