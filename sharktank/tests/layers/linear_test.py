# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch

from sharktank.layers import *
from sharktank.types import *


def _randomize_per_axis(t: torch.Tensor, axis: int, offset_range: float = 0.0):
    # Applies a randomized per-axis scale and offset to a tensor.
    bcast_shape = [1] * len(t.shape)
    bcast_shape[axis] = t.shape[axis]

    rnd_mult = torch.rand(bcast_shape, dtype=torch.float32)
    t = t * rnd_mult
    rnd_offset = torch.rand(bcast_shape, dtype=torch.float32) * offset_range
    return t + rnd_offset


def _scale_offset_per_axis_ui8(t: torch.Tensor, reduce_dim: int):
    mn, _ = torch.min(t, reduce_dim)
    mx, _ = torch.max(t, reduce_dim)
    scale = 255.0 / (mx - mn)
    offset = torch.round(mn * scale)
    return scale, offset.to(dtype=torch.uint8)


def _scale_per_tensor_i8(t: torch.Tensor):
    amax = torch.abs(torch.max(t))
    scale = 127 / amax.clamp(1e-6)
    return scale


class LinearQuantTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12345)

    def testNativeQuant_SymPerTensor_AsymPerAxis0_Dynamic(self):
        # Tests a linear layer that multiplies a per-tensor lhs with a
        # per-axis(0) rhs to produce a dynamically scaled FP result as output.

        # Generate random tensors that are then randomly scaled along axis-0.
        # Bias the rhs slightly to induce more interesting zero points.
        lhs = _randomize_per_axis(torch.rand(4, 8, 128, dtype=torch.float32), axis=0)
        rhs = _randomize_per_axis(
            torch.rand(16, 128, dtype=torch.float32), axis=0, offset_range=0.02
        )
        bias = torch.rand(16, dtype=torch.float32) + 5.0
        # bias = torch.zeros(16, dtype=torch.float32)

        lhs_scale = _scale_per_tensor_i8(lhs)
        rhs_scale, rhs_offset = _scale_offset_per_axis_ui8(rhs, 1)

        lhs_quantizer = StaticScaledQuantizer(
            name="q_input", scale=lhs_scale, dtype=torch.int8
        )
        rhs_quantizer = StaticScaledQuantizer(
            scale=rhs_scale, offset=rhs_offset, dtype=torch.uint8, axis=0
        )
        rhs_quant = rhs_quantizer.quantize(rhs, name="weight")

        # Sanity check that dequant'ing the RHS is roughly the same.
        # rhs_dequant = rhs_quant.unpack().dequant()
        # print("RHS_DIFF:", torch.abs(rhs_dequant - rhs))
        # print("RHS:", rhs)
        # print("RHS_DEQUANT:", rhs_dequant)
        # torch.testing.assert_close(rhs_dequant, rhs, atol=1e-1, rtol=1e-2)

        theta = Theta(
            [
                lhs_quantizer,
                rhs_quant,
                DefaultPrimitiveTensor(name="bias", data=bias),
            ]
        )
        linear = LinearLayer(theta)

        output = linear(lhs)
        output_ref = torch.matmul(lhs, rhs.T) + bias
        print(torch.abs(output - output_ref))
        torch.testing.assert_close(output, output_ref, atol=1e-1, rtol=1e-1)

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
