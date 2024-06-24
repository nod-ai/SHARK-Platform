# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch
import torch.nn.functional as F

from sharktank import ops
from sharktank.types import *


class EqualTest(unittest.TestCase):
    def testEqualTorchTensors(self):
        a = torch.rand(2, 3, dtype=torch.float32)
        b = torch.clone(a)
        assert ops.equal(a, b)
        assert ops.equal(b, a)

    def testNotEqualTorchTensors(self):
        a = torch.rand(2, 3, dtype=torch.float32)
        b = torch.clone(a)
        b[0, 0] += 1
        assert not ops.equal(a, b)
        assert not ops.equal(b, a)

    def testEqualTorchTensorAndPrimitiveTensor(self):
        a = torch.rand(2, 3, dtype=torch.float32)
        b = DefaultPrimitiveTensor(data=torch.clone(a))
        assert ops.equal(a, b)
        assert ops.equal(b, a)

    def testEqualTorchTensorAndPrimitiveTensor(self):
        a = torch.rand(2, 3, dtype=torch.float32)
        b = DefaultPrimitiveTensor(data=torch.clone(a))
        b.as_torch()[0, 0] += 1
        assert not ops.equal(a, b)
        assert not ops.equal(b, a)


class EmbeddingLookupTest(unittest.TestCase):
    def testTorchImplNoCast(self):
        t1 = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        t2 = torch.rand(10, 3, dtype=torch.float32)
        result = ops.embedding_lookup(t1, t2, torch.float32)
        expected = F.embedding(t1, t2)
        torch.testing.assert_close(result, expected)

    def testTorchImplCast(self):
        t1 = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        t2 = torch.rand(10, 3, dtype=torch.float16)
        result = ops.embedding_lookup(t1, t2, torch.float32)
        expected = F.embedding(t1, t2.to(torch.float32))
        torch.testing.assert_close(result, expected)

    def testPrimitiveTensorRhs(self):
        t1 = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        t2 = torch.rand(10, 3, dtype=torch.float16)
        t2_pt = DefaultPrimitiveTensor(data=t2)
        result = ops.embedding_lookup(t1, t2_pt, torch.float32)
        expected = F.embedding(t1, t2.to(torch.float32))
        torch.testing.assert_close(result, expected)

    def testQuantizedTensorRhs(self):
        # TODO: Implement me. Quantized embedding lookup NYI completely.
        ...


class MatmulTest(unittest.TestCase):
    def testMatchFail(self):
        # This is just using matmul as a victim to test that failure/exceptions
        # are properly raised when no override is found.
        with self.assertRaisesRegex(
            NotImplementedError,
            r"Overridable operator.+does not have an implementation for argument types:.+int.+int",
        ):
            ops.matmul(1, 2)

    @unittest.skip("https://github.com/nod-ai/sharktank/issues/44")
    def testTorchImplTransposedRHS(self):
        t1 = torch.rand(32, 16, dtype=torch.float32)
        t2 = torch.rand(48, 16, dtype=torch.float16)
        result = ops.matmul(t1, t2.T)
        expected = torch.matmul(t1, t2.T.to(torch.float32))
        torch.testing.assert_close(result, expected)
        self.assertIs(
            ops._registry._TEST_LAST_OP_DISPATCH,
            ops.custom_impls.matmul_mmtfp_tensor_tensor,
        )

    @unittest.skip("https://github.com/nod-ai/sharktank/issues/44")
    def testTorchImplNonTransposedRHS(self):
        t1 = torch.rand(32, 16, dtype=torch.float32)
        t2 = torch.rand(16, 48, dtype=torch.float16)
        result = ops.matmul(t1, t2)
        expected = torch.matmul(t1, t2.to(torch.float32))
        torch.testing.assert_close(result, expected)
        self.assertIsNot(
            ops._registry._TEST_LAST_OP_DISPATCH,
            ops.custom_impls.matmul_mmtfp_tensor_tensor,
        )

    @unittest.skip("https://github.com/nod-ai/sharktank/issues/44")
    def testTorchImplTransposedPrimitiveRHS(self):
        t1 = torch.rand(32, 16, dtype=torch.float32)
        t2 = torch.rand(48, 16, dtype=torch.float16)
        t2_pt = DefaultPrimitiveTensor(data=t2)
        result = ops.matmul(t1, t2_pt.T)
        expected = torch.matmul(t1, t2.T.to(torch.float32))
        torch.testing.assert_close(result, expected)
        self.assertIs(
            ops._registry._TEST_LAST_OP_DISPATCH,
            ops.custom_impls.matmul_mmtfp_tensor_tensor,
        )

    def testTorchImplTransposedQuantizedRHS_BlockScaledLayout(self):
        a_dtype = torch.float32
        d_dtype = torch.float32
        ref_dtype = torch.float32
        a = torch.rand([4, 16, 3200], dtype=a_dtype) * 64
        d = torch.rand([3200, 100, 1], dtype=d_dtype) * 64
        qs = (torch.rand([3200, 100, 32], dtype=ref_dtype) * 32.0).to(torch.int8)
        rhs_pqt = PlanarQuantizedTensor(
            shape=[3200, 3200], layout=BlockScaledLayout([3200, 3200], d, qs)
        )
        result = ops.matmul(a, rhs_pqt, transpose_rhs=True)
        # Just verifying dispatch. Numerics are tested at the kernel level.
        self.assertIs(
            ops._registry._TEST_LAST_OP_DISPATCH,
            ops.custom_impls.matmul_generic_tensor_block_scaled,
        )

    def testTorchImplTransposedQuantizedRHS_BlockScaledOffsetI4(self):
        a_dtype = torch.float32
        d_dtype = torch.float32
        ref_dtype = torch.float32
        a = torch.rand([4, 16, 3200], dtype=a_dtype) / 256.0
        d = torch.rand([3200, 100, 1], dtype=d_dtype) / 256.0
        qs = (torch.rand([3200, 100, 16], dtype=ref_dtype) * 255.0).to(torch.uint8)
        m = torch.rand([3200, 100, 1], dtype=d_dtype) + 16.0
        rhs_pqt = PlanarQuantizedTensor(
            shape=[3200, 3200],
            layout=BlockScaledI4Layout([3200, 3200], d, qs, m=m, signed=False),
        )
        result = ops.matmul(a, rhs_pqt, transpose_rhs=True)
        # Just verifying dispatch. Numerics are tested at the kernel level.
        self.assertIs(
            ops._registry._TEST_LAST_OP_DISPATCH,
            ops.custom_impls.matmul_generic_tensor_block_scaled_i4,
        )

    # TODO: mmt_super_block_scaled_offset_q4_unsigned


class PermuteTest(unittest.TestCase):
    def testPermute(self):
        torch_tensor = torch.rand(3, 4, 5, dtype=torch.float32)
        permutation = [1, 0, 2]
        primitive_tensor = DefaultPrimitiveTensor(data=torch_tensor)
        expected_result = torch.permute(torch_tensor, permutation)

        permuted_torch_tensor = ops.permute(torch_tensor, permutation)
        permuted_primitive_tensor = ops.permute(primitive_tensor, permutation)

        assert torch.equal(expected_result, permuted_torch_tensor)
        assert torch.equal(expected_result, permuted_primitive_tensor)

    def testTensorPropertyT(self):
        torch_tensor = torch.rand(3, 5, dtype=torch.float32)
        primitive_tensor = DefaultPrimitiveTensor(data=torch_tensor)
        assert torch.equal(torch_tensor.T, primitive_tensor.T)


class RmsNormTest(unittest.TestCase):
    def _ref(self, x, weight, epsilon):
        variance = x.pow(2).mean(-1, keepdim=True)
        output = x * torch.rsqrt(variance + epsilon)
        output = output * weight
        return output

    def testTorchImpl(self):
        t1 = torch.rand(16, 128, dtype=torch.float32)
        t2 = torch.rand(16, 128, dtype=torch.float32)
        result = ops.rms_norm(t1, t2, epsilon=1e-10)
        actual = self._ref(t1, t2, epsilon=1e-10)
        torch.testing.assert_close(actual, result)

    def testTorchPrimitiveWeightImpl(self):
        t1 = torch.rand(16, 128, dtype=torch.float32)
        t2 = torch.rand(16, 128, dtype=torch.float32)
        t2_pt = DefaultPrimitiveTensor(data=t2)
        result = ops.rms_norm(t1, t2_pt, epsilon=1e-10)
        actual = self._ref(t1, t2, epsilon=1e-10)
        torch.testing.assert_close(actual, result)

    # TODO: Quantized tensor


class TestOpExport(unittest.TestCase):
    """Tests that the machinery holds up under dynamo torch.export.

    Dynamo can be finicky with dynamism, and we've had trouble, so verify.
    """

    def testExport(self):
        a_dtype = torch.float32
        d_dtype = torch.float32
        ref_dtype = torch.float32
        a = torch.rand([4, 16, 3200], dtype=a_dtype) / 256.0
        d = torch.rand([3200, 100, 1], dtype=d_dtype) / 256.0
        qs = (torch.rand([3200, 100, 16], dtype=ref_dtype) * 255.0).to(torch.uint8)
        m = torch.rand([3200, 100, 1], dtype=d_dtype) + 16.0

        class MyModule(torch.nn.Module):
            def forward(self, a, d, qs, m):
                rhs_pqt = PlanarQuantizedTensor(
                    shape=[3200, 3200],
                    layout=BlockScaledI4Layout([3200, 3200], d, qs, m=m, signed=False),
                )
                result = ops.linear(a, rhs_pqt)
                return result

        my_module = MyModule()
        ep = torch.export.export(my_module, (a, d, qs, m))
        s = str(ep)
        self.assertIn("mmt_block_scaled_offset_q4_unsigned.default", s)


if __name__ == "__main__":
    unittest.main()
