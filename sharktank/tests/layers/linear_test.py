# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch

from sharktank.layers import *
from sharktank.types import *


class LinearQuantTest(unittest.TestCase):
    def testFakeQuantPerTensorDynamic(self):
        # TODO: Testing matmuls on unscaled random data like this produces
        # mis-behaving numerics due to outliers. Makes it hard to have a
        # sanity check.
        weight_raw = torch.randn(10, 20, dtype=torch.float32)
        weight = DynamicScaledQuantizer(dtype=torch.int8).quantize(
            weight_raw, name="weight"
        )
        theta = Theta(
            [
                DynamicScaledQuantizer(dtype=torch.int8, name="fq_input"),
                weight,
            ]
        )
        linear = LinearLayer(theta)

        self.assertEqual(linear.mode, "fq")
        x = torch.randn(5, 20, dtype=torch.float32)
        y = linear(x)
        y_ref = torch.matmul(x, weight_raw.t())
        print("Y:", y)
        print("Y_REF:", y_ref)

    def testFakeQuantPerTensorStatic(self):
        # TODO: Testing matmuls on unscaled random data like this produces
        # mis-behaving numerics due to outliers. Makes it hard to have a
        # sanity check.
        weight_raw = torch.randn(10, 20, dtype=torch.float32)
        weight = DynamicScaledQuantizer(dtype=torch.int8).quantize(
            weight_raw, name="weight"
        )
        theta = Theta(
            [
                StaticScaledQuantizer(
                    dtype=torch.int8, name="fq_input", scale=torch.tensor(25.6)
                ),
                StaticScaledQuantizer(
                    dtype=torch.int8, name="fq_output", scale=torch.tensor(20.6)
                ),
                weight,
            ]
        )
        linear = LinearLayer(theta)

        self.assertEqual(linear.mode, "fq")
        x = torch.randn(5, 20, dtype=torch.float32)
        y = linear(x)
        y_ref = torch.matmul(x, weight_raw.t())
        print("Y:", y)
        print("Y_REF:", y_ref)

    # TODO: Test bias.


if __name__ == "__main__":
    unittest.main()
