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
            (16, 100, 32, 8, 1e-3, 1e-5),
            (8, 8, 8, 8, 1e-3, 1e-5),
            (3, 17, 5, 7, 1e-3, 1e-5),
        ]
    )
    def testBS32(m, n, k, p, self, atol, rtol):
        a_type = torch.float32
        lowp_type = torch.int8
        b_type = torch.bool
        query = (torch.rand([m, k] , dtype=a_type) * 16).to(torch.int8)
        key = (torch.rand([n, k], dtype=a_type) * 16).to(torch.int8)
        value = (torch.rand([p, n], dtype=a_type) * 16).to(torch.int8)
        query_s = (torch.rand([m], dtype=a_type) * 16).to(torch.int8)
        key_s = (torch.rand([n], dtype=a_type) * 16).to(torch.int8)
        value_s = (torch.rand([p], dtype=a_type) * 16).to(torch.int8)
        query_zp = (torch.rand([m], dtype=a_type) * 16).to(torch.int8)
        key_zp = (torch.rand([m], dtype=a_type) * 16).to(torch.int8)
        value_zp = (torch.rand([p], dtype=a_type) * 16).to(torch.int8)
        attn_mask = torch.rand([m,n], dtype=torch.float32) > .5
        randoms = torch.rand([m,n], dtype=a_type)
        dropout_p = torch.tensor(0, dtype=a_type) # for testing
        is_causal = torch.tensor(False) # true
        scale = torch.tensor(True) # true # todo, allow values
        
        result = kernels.attn_q8(query, key, value, query_s, key_s, value_s, query_zp, key_zp, value_zp, attn_mask, randoms, dropout_p, is_causal, scale)

        # Dequantize and test with normal matmul.
        # Tolerances are empirical and results are not expected to match exactly.
        q = query.to(a_type)
        k = key.to(a_type)
        v = value.to(a_type)
        for i in range(query.shape[0]):
            q[i] = (query[i].to(a_type) - query_zp[i].to(a_type)) * query_s[i].to(a_type)
        for i in range(key.shape[0]):
            k[i] = (key[i].to(a_type) - key_zp[i].to(a_type)) * key_s[i].to(a_type)
        for i in range(value.shape[0]):
            v[i] = (value[i].to(a_type) - value_zp[i].to(a_type)) * value_s[i].to(a_type)
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v.T, attn_mask=attn_mask, dropout_p=dropout_p.item(), is_causal=is_causal.item(), scale=1)
        torch.testing.assert_close(result, ref, atol=atol, rtol=rtol)

    def testExportStaticDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, query, key, value, query_s, key_s, value_s, query_zp, key_zp, value_zp, attn_mask, randoms, dropout_p, is_causal, scale):
                return kernels.attn_q8(query, key, value, query_s, key_s, value_s, query_zp, key_zp, value_zp, attn_mask, randoms, dropout_p, is_causal, scale)

        mod = MyModule()
        a_type = torch.float32
        lowp_type = torch.int8
        b_type = torch.bool
        ep = torch.export.export(
            mod,
            args=(
                (torch.rand([16, 32], dtype=a_type) * 16).to(torch.int8),
                (torch.rand([100, 32], dtype=a_type) * 16).to(torch.int8),
                (torch.rand([8, 100], dtype=a_type) * 16).to(torch.int8),
                (torch.rand([16], dtype=a_type) * 16).to(torch.int8),
                (torch.rand([100], dtype=a_type) * 16).to(torch.int8),
                (torch.rand([8], dtype=a_type) * 16).to(torch.int8),
                (torch.rand([16], dtype=a_type) * 16).to(torch.int8),
                (torch.rand([100], dtype=a_type) * 16).to(torch.int8),
                (torch.rand([8], dtype=a_type) * 16).to(torch.int8),
                torch.rand([16,100], dtype=torch.float32) > .5,
                torch.rand([16,100], dtype=a_type),
                torch.tensor(0, dtype=a_type), # for testing
                torch.tensor(False),
                torch.tensor(True), # todo, allow values

            ),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn("@attn_q8", asm)


if __name__ == "__main__":
    unittest.main()
