# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest
from parameterized import parameterized

import torch

from shark_turbine import aot
from sharktank import kernels


def _pad_last_2d(input_tensor, pad_width):
    # pad_width should be in the format [pad_left, pad_right, pad_top, pad_bottom]
    pad_left, pad_right, pad_top, pad_bottom = pad_width
    batch_size, channels, height, width = input_tensor.shape

    # Create a new tensor with the desired padded size filled with zeros
    padded_height = height + pad_top + pad_bottom
    padded_width = width + pad_left + pad_right
    padded_tensor = torch.zeros(
        (batch_size, channels, padded_height, padded_width), dtype=input_tensor.dtype
    )

    # Copy the values from the input tensor to the appropriate location in the padded tensor
    padded_tensor[
        :, :, pad_top : pad_top + height, pad_left : pad_left + width
    ] = input_tensor
    return padded_tensor


class conv_2d_nchw_fchw_test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    @parameterized.expand(
        [
            (1e-3, 1e-5),
            (1e-3, 1e-5),
            (1e-3, 1e-5),
        ]
    )
    def testBS32(self, atol, rtol):
        dtype = torch.int8
        inputs = (torch.rand([2, 320, 64, 64]) * 64).to(dtype)
        padding = [1, 1]
        extended_list = [item for item in padding for _ in range(2)]
        inputs_pad = _pad_last_2d(inputs, extended_list)
        weights = (torch.rand([640, 320, 3, 3]) * 64).to(dtype)
        bias = (torch.rand([640]) * 64).to(dtype)
        result = kernels.conv_2d_nchw_fchw(
            inputs, inputs_pad, weights, bias, [1, 1], [1, 1], [1, 1]
        )

        # Tolerances are empirical and results are not expected to match exactly.
        ref = torch.nn.functional.conv2d(
            inputs, weights, bias=bias, stride=(1, 1), padding=(1, 1), dilation=(1, 1)
        )
        print(result.shape)
        print(ref.shape)
        torch.testing.assert_close(result, ref, atol=atol, rtol=rtol)

    def testExportStaticDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, a, b, c, d):
                return kernels.conv_2d_nchw_fchw(a, b, c, d, [1, 1], [1, 1], [1, 1])

        mod = MyModule()
        dtype = torch.int8
        inputs = torch.rand([2, 320, 64, 64]) * 64
        padding = [1, 1]
        extended_list = [item for item in padding for _ in range(2)]
        inputs_pad = _pad_last_2d(inputs, extended_list)
        ep = torch.export.export(
            mod,
            args=(
                (inputs).to(dtype),
                (inputs_pad).to(dtype),
                (torch.rand([640, 320, 3, 3]) * 64).to(dtype),
                (torch.rand([640]) * 64).to(dtype),
            ),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn("@sharktank_conv_2d_nchw_fchw", asm)


if __name__ == "__main__":
    unittest.main()
