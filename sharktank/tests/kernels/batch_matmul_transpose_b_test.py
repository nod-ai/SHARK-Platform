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


class batch_matmul_transpose_b_test(unittest.TestCase):
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
        a = (torch.rand([4, 16, 3200]) * 64).to(dtype)
        b = (torch.rand([4, 8, 3200]) * 64).to(dtype)
        result = kernels.batch_matmul_transpose_b(a, b)

        # Dequantize and test with normal matmul.
        # Tolerances are empirical and results are not expected to match exactly.
        bT = torch.transpose(b, 1, 2)
        ref = torch.matmul(a, bT)
        torch.testing.assert_close(result, ref, atol=atol, rtol=rtol)

    def testExportStaticDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, a, b):
                return kernels.batch_matmul_transpose_b(a, b)

        mod = MyModule()
        dtype = torch.int32
        ep = torch.export.export(
            mod,
            args=(
                (torch.rand([4, 16, 2]) * 64).to(dtype),
                (torch.rand([4, 8, 2]) * 64).to(dtype),
            ),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn("@sharktank_batch_matmul_transpose_b_16_8_2_i32", asm)


if __name__ == "__main__":
    unittest.main()
