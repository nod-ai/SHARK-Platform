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


class pooling_nchw_sum_test(unittest.TestCase):
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
        dtype = torch.int32
        a = (torch.randint(0, 100, (2, 1, 128, 128))).to(torch.float32)
        weight_shape = [3, 3]
        stride = [1, 1]
        padding = [1, 1]
        dilations = [1, 1]
        result = kernels.pooling_nchw_sum(
            a.to(dtype), weight_shape, stride, padding, dilations
        )

        # Tolerances are empirical and results are not expected to match exactly.
        ref = torch.nn.functional.avg_pool2d(
            a,
            weight_shape,
            stride=stride,
            padding=padding,
            divisor_override=1,
        ).to(dtype)
        torch.testing.assert_close(result, ref, atol=atol, rtol=rtol)

    def testExportStaticDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, a):
                return kernels.pooling_nchw_sum(a, [3, 3], [1, 1], [1, 1], [1, 1])

        mod = MyModule()
        dtype = torch.int32
        ep = torch.export.export(
            mod,
            args=((torch.rand([2, 1, 128, 128]) * 64).to(dtype),),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        print(asm)
        self.assertIn("@sharktank_pooling_nchw_sum", asm)


if __name__ == "__main__":
    unittest.main()
