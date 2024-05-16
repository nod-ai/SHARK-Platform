# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch

from sharktank import ops
from sharktank.types import *


class MatmulTest(unittest.TestCase):
    def testTorchRHSColumnShardedTransposed(self):
        t1 = torch.rand(4, 32, 16, dtype=torch.float32)
        t2 = torch.rand(48, 16, dtype=torch.float16)
        # RHS is transposed, so dim0 is the "column". Shard into 12.
        t2_sharded = ShardedPrimitiveTensor(shard_dim=0, ts=t2.split(4, dim=0))
        sharded_result = ops.matmul(t1, t2_sharded)
        expected_result = ops.matmul(t1, t2)
        unsharded_result = ops.sharded_cat(sharded_result)
        torch.testing.assert_close(unsharded_result, expected_result)

    def testTorchRHSColumnSharded(self):
        t1 = torch.rand(4, 32, 16, dtype=torch.float32)
        t2 = torch.rand(16, 48, dtype=torch.float16)
        t2_sharded = ShardedPrimitiveTensor(shard_dim=1, ts=t2.split(4, dim=1))
        sharded_result = ops.matmul(t1, t2_sharded, transpose_rhs=False)
        expected_result = ops.matmul(t1, t2, transpose_rhs=False)
        unsharded_result = ops.sharded_cat(sharded_result)
        torch.testing.assert_close(unsharded_result, expected_result)

    def testShardedChainMatmulX2Transposed(self):
        # Computes Z = (XA)B (sharded by 8).
        X = torch.rand(4, 32, 16, dtype=torch.float32)
        A = torch.rand(48, 16, dtype=torch.float16)
        B = torch.rand(16, 48, dtype=torch.float16)
        XA = ops.matmul(X, A)
        Z = ops.matmul(XA, B)

        # Columnwise sharding of A matrix (transposed).
        A_sharded = ShardedPrimitiveTensor(shard_dim=0, ts=A.split(6, dim=0))
        assert A_sharded.shard_count == 8
        # Rowwise sharding of B matrix (transposed).
        B_sharded = ShardedPrimitiveTensor(shard_dim=1, ts=B.split(6, dim=1))
        assert B_sharded.shard_count == 8

        XA_sharded = ops.matmul(X, A_sharded)
        Z_sharded = ops.matmul(XA_sharded, B_sharded)
        Z_unsharded = ops.sharded_sum(Z_sharded)
        torch.testing.assert_close(Z_unsharded, Z)

    def testShardedFFNTransposed(self):
        input = torch.rand(4, 32, 64, dtype=torch.float32)
        unsharded_ffn_gate_weight = torch.rand(128, 64, dtype=torch.float16)
        unsharded_ffn_down_weight = torch.rand(64, 128, dtype=torch.float16)
        unsharded_ffn_up_weight = torch.rand(128, 64, dtype=torch.float16)

        def compute(input, ffn_gate_weight, ffn_down_weight, ffn_up_weight):
            ffn_gate = ops.elementwise(
                torch.nn.functional.silu, ops.matmul(input, ffn_gate_weight)
            )
            ffn_up = ops.matmul(input, ffn_up_weight)
            ffn_down = ops.matmul(
                ops.elementwise(torch.mul, ffn_gate, ffn_up), ffn_down_weight
            )
            summed = ops.sharded_sum(ffn_down)
            return summed

        Z_ref = compute(
            input,
            unsharded_ffn_gate_weight,
            unsharded_ffn_down_weight,
            unsharded_ffn_up_weight,
        )

        # Columnwise sharding of gate and up weight (transposed).
        sharded_ffn_gate_weight = ShardedPrimitiveTensor(
            shard_dim=0, ts=unsharded_ffn_gate_weight.split(16, dim=0)
        )
        sharded_ffn_up_weight = ShardedPrimitiveTensor(
            shard_dim=0, ts=unsharded_ffn_up_weight.split(16, dim=0)
        )
        assert sharded_ffn_gate_weight.shard_count == 8
        assert sharded_ffn_up_weight.shard_count == 8

        # Rowwise sharding of down weight (transposed).
        sharded_ffn_down_weight = ShardedPrimitiveTensor(
            shard_dim=1, ts=unsharded_ffn_down_weight.split(16, dim=1)
        )
        assert sharded_ffn_down_weight.shard_count == 8
        Z_sharded = compute(
            input,
            sharded_ffn_gate_weight,
            sharded_ffn_down_weight,
            sharded_ffn_up_weight,
        )
        torch.testing.assert_close(Z_sharded, Z_ref)


if __name__ == "__main__":
    unittest.main()
