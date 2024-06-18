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
        dtype = torch.int32
        a = (torch.rand([4,6,8,10]) * 64).to(dtype)
        b = (torch.rand([6,6,1,1]) * 64).to(dtype)
        result = kernels.conv_2d_nchw_fchw(a, b, [1, 1], [1, 1], [1, 1])

        # Tolerances are empirical and results are not expected to match exactly.
        ref = torch.nn.functional.conv2d(a, b, stride=(1,1), padding=1, dilation=(1,1))
        print(result)
        print(ref)
        torch.testing.assert_close(result, ref, atol=atol, rtol=rtol)

"""    def testExportStaticDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, a, b, c, d, e):
                return kernels.conv_2d_nchw_fchw(a, b, c, d, e)

        mod = MyModule()
        dtype = torch.int32
        ep = torch.export.export(
            mod,
            args=(
                (torch.rand([8,8,8,8]) * 64).to(dtype),
                (torch.rand([8,8,1,1]) * 64).to(dtype),
                "1, 1", "0, 0", "1, 1"
            ),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn("@sharktank_conv_2d_nchw_fchw", asm)
"""

if __name__ == "__main__":
    unittest.main()
