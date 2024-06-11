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


class mmt_scaled_q8_test(unittest.TestCase):
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
        ref_dtype = torch.float32
        lhs_dtype = torch.int8
        rhs_dype = torch.int8
        lhs = (torch.rand([16, 3200], dtype=ref_dtype) * 32.0).to(torch.int8)
        rhs = (torch.rand([100, 3200], dtype=ref_dtype) * 32.0).to(torch.int8)
        scale0 = (torch.rand(tuple(), dtype=ref_dtype) * 32.0).to(torch.int8)
        scale1 = (torch.rand(tuple(), dtype=ref_dtype) * 32.0).to(torch.int8)
        result = kernels.mmt_scaled_q8(lhs, rhs, scale0, scale1)

        # Dequantize and test with normal matmul.
        # Tolerances are empirical and results are not expected to match exactly.
        scaled_lhs = (lhs.to(ref_dtype) * scale0.to(ref_dtype))
        scaled_rhs = (rhs.to(ref_dtype) * scale1.to(ref_dtype))
        ref = torch.matmul(scaled_lhs.to(ref_dtype), scaled_rhs.T.to(ref_dtype))
        torch.testing.assert_close(result, ref, atol=atol, rtol=rtol)

    def testExportStaticDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, lhs, rhs, scale0, scale1):
                return kernels.mmt_scaled_q8(lhs, rhs, scale0, scale1)

        mod = MyModule()
        ep = torch.export.export(
            mod,
            args=(
                (torch.rand([16, 3200], dtype=torch.float32) * 32.0).to(torch.int8),
                (torch.rand([100, 3200], dtype=torch.float32) * 32.0).to(torch.int8),
                (torch.rand(tuple(), dtype=torch.float32) * 32.0).to(
                    torch.int8
                ),
                (torch.rand(tuple(), dtype=torch.float32) * 32.0).to(
                    torch.int8
                ),

            ),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn("@mmt_scaled_q8", asm)


if __name__ == "__main__":
    unittest.main()
