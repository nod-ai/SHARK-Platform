# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from parameterized import parameterized

import torch

from sharktank import ops
from sharktank.types import *


class ConvTest(unittest.TestCase):
    def testAllGather(self):
        shard_count = 3
        shard_shape = [3, 4]
        shard_dim = 1
        shards = [
            torch.rand(shard_shape, dtype=torch.float32) for i in range(shard_count)
        ]
        expected_result = torch.cat(shards, dim=shard_dim)

        sharded = SplitPrimitiveTensor(shard_dim=shard_dim, ts=shards)
        actual_result = ops.all_gather(sharded)

        for shard in actual_result.shards:
            torch.testing.assert_close(shard.as_torch(), expected_result)

    def testConv2dShardedInputAndOutputChannelsOneGroup(self):
        batches = 2
        in_channels = 6
        out_channels = 12
        groups = 1
        height = 17
        width = 19
        stride = 2
        padding = 3
        dilation = 2
        kernel_height = 3
        kernel_width = 4
        x = torch.rand(batches, in_channels, height, width, dtype=torch.float32)
        weight = torch.rand(
            out_channels,
            in_channels // groups,
            kernel_height,
            kernel_width,
            dtype=torch.float32,
        )
        bias = torch.rand(out_channels, dtype=torch.float32)

        expected_result = ops.conv2d(
            x,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        shard_count = 2
        x_sharded = SplitPrimitiveTensor(shard_dim=1, ts=x, shard_count=shard_count)
        weight_sharded = SplitPrimitiveTensor(
            shard_dim=0, ts=weight, shard_count=shard_count
        )
        bias_sharded = SplitPrimitiveTensor(
            shard_dim=0, ts=bias, shard_count=shard_count
        )
        sharded_result = ops.conv2d(
            x,
            weight=weight_sharded,
            bias=bias_sharded,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        actual_result = ops.sharded_cat(sharded_result)

        torch.testing.assert_close(actual_result, expected_result)

    def testCov2dShardedOutputChannelsOneGroup(self):
        batches = 2
        in_channels = 6
        out_channels = 12
        groups = 1
        height = 17
        width = 19
        stride = 2
        padding = 3
        dilation = 2
        kernel_height = 3
        kernel_width = 4
        x = torch.rand(batches, in_channels, height, width, dtype=torch.float32)
        weight = torch.rand(
            out_channels,
            in_channels // groups,
            kernel_height,
            kernel_width,
            dtype=torch.float32,
        )
        bias = torch.rand(out_channels, dtype=torch.float32)

        expected_result = ops.conv2d(
            x,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        shard_count = 2
        weight_sharded = SplitPrimitiveTensor(
            shard_dim=0, ts=weight, shard_count=shard_count
        )
        bias_sharded = SplitPrimitiveTensor(
            shard_dim=0, ts=bias, shard_count=shard_count
        )
        sharded_result = ops.conv2d(
            x,
            weight=weight_sharded,
            bias=bias_sharded,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        actual_result = ops.sharded_cat(sharded_result)

        torch.testing.assert_close(actual_result, expected_result)


class ElementwiseTest(unittest.TestCase):
    def testRhsAndLhsShardedAdd(self):
        a = torch.rand(4, 5, 6, dtype=torch.float32)
        b = torch.rand(4, 5, 6, dtype=torch.float32)

        expected_result = a + b

        shard_dim = 2
        shard_count = 3
        sharded_a = ops.reshard_split(a, dim=shard_dim, count=shard_count)
        sharded_b = ops.reshard_split(b, dim=shard_dim, count=shard_count)
        sharded_result = sharded_a + sharded_b
        actual_result = ops.reshard_like(sharded_result, expected_result)

        torch.testing.assert_close(actual_result, expected_result)

    @parameterized.expand(
        [
            (torch.add,),
            (torch.div,),
            (torch.fmin,),
            (torch.fmax,),
            (torch.sub),
        ]
    )
    def testBinaryOperators(self, operator):
        a = torch.rand(4, 5, 6, dtype=torch.float32)
        b = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_dim = 2
        shard_count = 3
        expected_result = operator(a, b)

        # Sharded LHS and RHS
        sharded_a = ops.reshard_split(a, dim=shard_dim, count=shard_count)
        sharded_b = ops.reshard_split(b, dim=shard_dim, count=shard_count)
        sharded_result = ops.elementwise(operator, sharded_a, sharded_b)
        assert isinstance(sharded_result, ShardedTensor)
        assert not sharded_result.is_replicated
        assert sharded_result.shard_count == sharded_a.shard_count
        assert sharded_result.shard_dim == sharded_a.shard_dim
        actual_result = ops.reshard_like(sharded_result, expected_result)
        torch.testing.assert_close(actual_result, expected_result)

        # Replicated LHS and Sharded RHS
        sharded_a = ops.replicate(a, count=shard_count)
        sharded_b = ops.reshard_split(b, dim=shard_dim, count=shard_count)
        sharded_result = ops.elementwise(operator, sharded_a, sharded_b)
        assert isinstance(sharded_result, ShardedTensor)
        assert not sharded_result.is_replicated
        assert sharded_result.shard_count == sharded_b.shard_count
        assert sharded_result.shard_dim == sharded_b.shard_dim
        actual_result = ops.reshard_like(sharded_result, expected_result)
        torch.testing.assert_close(actual_result, expected_result)

        # Sharded LHS and Replicated RHS
        sharded_a = ops.reshard_split(a, dim=shard_dim, count=shard_count)
        sharded_b = ops.replicate(b, count=shard_count)
        sharded_result = ops.elementwise(operator, sharded_a, sharded_b)
        assert isinstance(sharded_result, ShardedTensor)
        assert not sharded_result.is_replicated
        assert sharded_result.shard_count == sharded_a.shard_count
        assert sharded_result.shard_dim == sharded_a.shard_dim
        actual_result = ops.reshard_like(sharded_result, expected_result)
        torch.testing.assert_close(actual_result, expected_result)


class EqualTest(unittest.TestCase):
    def testNotEqualReplicated(self):
        a = torch.rand(3, 4, 5, dtype=torch.float32)
        b = torch.clone(a)
        shard_count = 2
        a_sharded = ops.replicate(a, count=shard_count)
        b_sharded = ops.reshard_like(b, a_sharded)
        assert ops.equal(a_sharded, b_sharded)
        assert ops.equal(b_sharded, a_sharded)

    def testNotEqualReplicated(self):
        a = torch.rand(3, 4, 5, dtype=torch.float32)
        b = torch.clone(a)
        b[0, 0, 0] += 1
        shard_count = 2
        a_sharded = ops.replicate(a, count=shard_count)
        b_sharded = ops.reshard_like(b, a_sharded)
        assert not ops.equal(a_sharded, b_sharded)
        assert not ops.equal(b_sharded, a_sharded)

    def testEqualSharded(self):
        a = torch.rand(3, 4, 5, dtype=torch.float32)
        b = torch.clone(a)
        shard_dim = 1
        shard_count = 2
        a_sharded = ops.reshard_split(a, dim=shard_dim, count=shard_count)
        b_sharded = ops.reshard_like(b, a_sharded)
        assert ops.equal(a_sharded, b_sharded)
        assert ops.equal(b_sharded, a_sharded)

    def testNotEqualSharded(self):
        a = torch.rand(3, 4, 5, dtype=torch.float32)
        b = torch.clone(a)
        b[0, 0, 0] += 1
        shard_dim = 1
        shard_count = 2
        a_sharded = ops.reshard_split(a, dim=shard_dim, count=shard_count)
        b_sharded = ops.reshard_like(b, a_sharded)
        assert not ops.equal(a_sharded, b_sharded)
        assert not ops.equal(b_sharded, a_sharded)


class NormalizationTest(unittest.TestCase):
    def testGroupNormShardedGroups(self):
        """Shard the channel dimension such that the group count is multiple of the
        shard count."""
        batches = 3
        groups = 6
        height = 17
        width = 19
        channels = 12
        eps = 0.01
        x = torch.rand(batches, channels, height, width, dtype=torch.float32)
        weight = torch.rand(channels, dtype=torch.float32)
        bias = torch.rand(channels, dtype=torch.float32)

        expected_result = ops.group_norm_affine(
            x, weight=weight, bias=bias, num_groups=groups, eps=eps
        )

        shard_count = 3
        x_sharded = SplitPrimitiveTensor(shard_dim=1, ts=x, shard_count=shard_count)
        weight_sharded = SplitPrimitiveTensor(
            shard_dim=0, ts=weight, shard_count=shard_count
        )
        bias_sharded = SplitPrimitiveTensor(
            shard_dim=0, ts=bias, shard_count=shard_count
        )
        sharded_result = ops.group_norm_affine(
            x_sharded,
            weight=weight_sharded,
            bias=bias_sharded,
            num_groups=groups,
            eps=eps,
        )
        actual_result = ops.sharded_cat(sharded_result)

        torch.testing.assert_close(actual_result, expected_result)

    def testLayerNorm(self):
        """Shard an input dimension other than the trailing normalization dimensions."""
        batches = 3
        eps = 0.01
        weight = torch.rand(3, 4, dtype=torch.float32)
        bias = torch.rand_like(weight)
        input_shape = [batches, 11, 12] + list(weight.shape)
        x = torch.rand(input_shape, dtype=torch.float32)

        expected_result = ops.layer_norm(x, weight=weight, bias=bias, eps=eps)

        x_sharded = SplitPrimitiveTensor(shard_dim=2, ts=x, shard_count=3)
        sharded_result = ops.layer_norm(
            x_sharded,
            weight=weight,
            bias=bias,
            eps=eps,
        )
        actual_result = ops.sharded_cat(sharded_result)

        torch.testing.assert_close(actual_result, expected_result)


class PermuteTest(unittest.TestCase):
    def testShardedPrimitiveTensorPermute(self):
        torch_tensor = torch.rand(3, 8, 5, dtype=torch.float32)
        permutation = [1, 0, 2]
        sharded_tensor = SplitPrimitiveTensor(
            ts=torch_tensor, shard_dim=1, shard_count=4
        )
        expected_result = torch.permute(torch_tensor, permutation)

        permuted_sharded_tensor = ops.permute(sharded_tensor, permutation)
        result = ops.sharded_cat(permuted_sharded_tensor)

        assert ops.equal(expected_result, result)


class MatmulTest(unittest.TestCase):
    def testTorchRHSColumnShardedTransposed(self):
        t1 = torch.rand(4, 32, 16, dtype=torch.float32)
        t2 = torch.rand(48, 16, dtype=torch.float16)
        # RHS is transposed, so dim0 is the "column". Shard into 12.
        t2_sharded = SplitPrimitiveTensor(shard_dim=0, ts=t2.split(4, dim=0))
        sharded_result = ops.matmul(t1, t2_sharded.T)
        expected_result = ops.matmul(t1, t2.T)
        unsharded_result = ops.sharded_cat(sharded_result)
        torch.testing.assert_close(unsharded_result, expected_result)

    def testTorchRHSColumnSharded(self):
        t1 = torch.rand(4, 32, 16, dtype=torch.float32)
        t2 = torch.rand(16, 48, dtype=torch.float16)
        t2_sharded = SplitPrimitiveTensor(shard_dim=1, ts=t2.split(4, dim=1))
        sharded_result = ops.matmul(t1, t2_sharded)
        expected_result = ops.matmul(t1, t2)
        unsharded_result = ops.sharded_cat(sharded_result)
        torch.testing.assert_close(unsharded_result, expected_result)

    def testShardedChainMatmulX2Transposed(self):
        # Computes Z = (XA)B (sharded by 8).
        X = torch.rand(4, 32, 16, dtype=torch.float32)
        A = torch.rand(48, 16, dtype=torch.float16)
        B = torch.rand(16, 48, dtype=torch.float16)
        XA = ops.matmul(X, A.T)
        Z = ops.matmul(XA, B.T)

        # Columnwise sharding of A matrix (transposed).
        A_sharded = SplitPrimitiveTensor(shard_dim=0, ts=A.split(6, dim=0))
        assert A_sharded.shard_count == 8
        # Rowwise sharding of B matrix (transposed).
        B_sharded = SplitPrimitiveTensor(shard_dim=1, ts=B.split(6, dim=1))
        assert B_sharded.shard_count == 8

        XA_sharded = ops.matmul(X, A_sharded.T)
        Z_sharded = ops.matmul(XA_sharded, B_sharded.T)
        Z_unsharded = ops.sharded_sum(Z_sharded)
        torch.testing.assert_close(Z_unsharded, Z)

    def testShardedParallelAxisInLhs(self):
        a = torch.rand(2, 12, 5, dtype=torch.float32)
        b = torch.rand(5, 9, dtype=torch.float32)
        expected_result = torch.matmul(a, b)
        shard_count = 3
        a_sharded = SplitPrimitiveTensor(ts=a, shard_dim=1, shard_count=shard_count)
        res_sharded = ops.matmul(a_sharded, b)
        assert isinstance(res_sharded, SplitPrimitiveTensor)
        assert res_sharded.shard_dim == 1
        assert res_sharded.shard_count == shard_count
        actual_result = ops.sharded_cat(res_sharded)
        torch.testing.assert_close(actual_result, expected_result)

    def testShardedParallelAxesInLhsAndRhs(self):
        a = torch.rand(2, 12, 5, dtype=torch.float32)
        b = torch.rand(5, 9, dtype=torch.float32)
        expected_result = torch.matmul(a, b)
        shard_count = 3
        a_sharded = SplitPrimitiveTensor(ts=a, shard_dim=1, shard_count=shard_count)
        b_sharded = SplitPrimitiveTensor(ts=b, shard_dim=1, shard_count=shard_count)
        res_sharded = ops.matmul(a_sharded, b_sharded)
        assert isinstance(res_sharded, SplitPrimitiveTensor)
        assert res_sharded.shard_dim == 1
        assert res_sharded.shard_count == shard_count
        actual_result = ops.sharded_cat(res_sharded)
        torch.testing.assert_close(actual_result, expected_result)

    def testShardedParallelAxesInLhsAndTransposedRhs(self):
        a = torch.rand(2, 12, 5, dtype=torch.float32)
        b = torch.rand(9, 5, dtype=torch.float32)
        expected_result = torch.matmul(a, b.T)
        shard_count = 3
        a_sharded = SplitPrimitiveTensor(ts=a, shard_dim=1, shard_count=shard_count)
        b_sharded = SplitPrimitiveTensor(ts=b, shard_dim=0, shard_count=shard_count)
        res_sharded = ops.matmul(a_sharded, b_sharded, transpose_rhs=True)
        assert isinstance(res_sharded, SplitPrimitiveTensor)
        assert res_sharded.shard_dim == 1
        assert res_sharded.shard_count == shard_count
        actual_result = ops.sharded_cat(res_sharded)
        torch.testing.assert_close(actual_result, expected_result)

    def testShardedLhsBatchDimAndRhsParallelDim(self):
        a = torch.rand(12, 2, 5, dtype=torch.float32)
        b = torch.rand(5, 9, dtype=torch.float32)
        expected_result = torch.matmul(a, b)
        shard_count = 3
        a_sharded = SplitPrimitiveTensor(ts=a, shard_dim=0, shard_count=shard_count)
        b_sharded = SplitPrimitiveTensor(ts=b, shard_dim=1, shard_count=shard_count)
        res_sharded = ops.matmul(a_sharded, b_sharded)
        assert isinstance(res_sharded, SplitPrimitiveTensor)
        assert res_sharded.shard_dim == 0
        assert res_sharded.shard_count == shard_count
        actual_result = ops.sharded_cat(res_sharded)
        torch.testing.assert_close(actual_result, expected_result)

    def testShardedLhsReplcatedRhs(self):
        a = torch.rand(12, 3, 5, dtype=torch.float32)
        b = torch.rand(5, 9, dtype=torch.float32)
        expected_result = torch.matmul(a, b)
        shard_count = 3
        a_sharded = SplitPrimitiveTensor(ts=a, shard_dim=1, shard_count=shard_count)
        b_sharded = ReplicatedTensor(ts=b, shard_count=shard_count)
        res_sharded = ops.matmul(a_sharded, b_sharded)
        assert isinstance(res_sharded, SplitPrimitiveTensor)
        assert res_sharded.shard_dim == 1
        assert res_sharded.shard_count == shard_count
        actual_result = ops.sharded_cat(res_sharded)
        torch.testing.assert_close(actual_result, expected_result)

    def testShardedFFNTransposed(self):
        input = torch.rand(4, 32, 64, dtype=torch.float32)
        unsharded_ffn_gate_weight = torch.rand(128, 64, dtype=torch.float16)
        unsharded_ffn_down_weight = torch.rand(64, 128, dtype=torch.float16)
        unsharded_ffn_up_weight = torch.rand(128, 64, dtype=torch.float16)

        def compute(input, ffn_gate_weight, ffn_down_weight, ffn_up_weight):
            ffn_gate = ops.elementwise(
                torch.nn.functional.silu, ops.linear(input, ffn_gate_weight)
            )
            ffn_up = ops.linear(input, ffn_up_weight)
            ffn_down = ops.linear(
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
        sharded_ffn_gate_weight = SplitPrimitiveTensor(
            shard_dim=0, ts=unsharded_ffn_gate_weight.split(16, dim=0)
        )
        sharded_ffn_up_weight = SplitPrimitiveTensor(
            shard_dim=0, ts=unsharded_ffn_up_weight.split(16, dim=0)
        )
        assert sharded_ffn_gate_weight.shard_count == 8
        assert sharded_ffn_up_weight.shard_count == 8

        # Rowwise sharding of down weight (transposed).
        sharded_ffn_down_weight = SplitPrimitiveTensor(
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


class ReplicateTest(unittest.TestCase):
    def testReplicateReplicated(self):
        tensor = torch.rand(4, 5, dtype=torch.float32)
        shard_count = 3
        expected_result = ops.replicate(tensor, count=shard_count)
        actual_result = ops.replicate(expected_result, count=shard_count)
        assert expected_result.is_deep_equal(actual_result)

    def testReplicateUnsharded(self):
        tensor = torch.rand(4, 5, dtype=torch.float32)
        shard_count = 3
        actual_result = ops.replicate(tensor, count=shard_count)
        expected_result = ReplicatedTensor(ts=tensor, shard_count=shard_count)
        assert expected_result.is_deep_equal(actual_result)


class ReshardTest(unittest.TestCase):
    def testReshardReplicated(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_dim = 2
        shard_count = 3
        replicated_tensor = ops.replicate(tensor, count=shard_count)
        actual_result = ops.reshard_split(
            replicated_tensor, dim=shard_dim, count=shard_count
        )
        expected_result = ops.reshard_split(tensor, dim=shard_dim, count=shard_count)
        assert expected_result.is_deep_equal(actual_result)

    def testReshardUnsharded(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_dim = 2
        shard_count = 3
        actual_result = ops.reshard_split(tensor, dim=shard_dim, count=shard_count)
        expected_result = SplitPrimitiveTensor(
            ts=tensor, shard_count=shard_count, shard_dim=shard_dim
        )
        assert expected_result.is_deep_equal(actual_result)

    def testReshardSharded(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_dim = 2
        shard_count = 3
        expected_result = SplitPrimitiveTensor(
            ts=tensor, shard_count=shard_count, shard_dim=shard_dim
        )
        actual_result = ops.reshard_split(
            expected_result, dim=shard_dim, count=shard_count
        )
        assert expected_result.is_deep_equal(actual_result)


class ShardLikeTest(unittest.TestCase):
    def testReshardLikeReplicatedToReplicated(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_count = 2
        expected_result = ops.replicate(tensor, count=shard_count)
        actual_result = ops.reshard_like(expected_result, expected_result)
        assert expected_result.is_deep_equal(actual_result)

    def testReshardLikeReplicatedToSharded(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_dim = 2
        shard_count = 3
        expected_result = ops.reshard_split(tensor, dim=shard_dim, count=shard_count)
        replicated_tensor = ops.replicate(tensor, count=shard_count)
        actual_result = ops.reshard_like(replicated_tensor, expected_result)
        assert expected_result.is_deep_equal(actual_result)

    def testReshardLikeReplicatedToUnsharded(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_count = 2
        replicated = ops.replicate(tensor, count=shard_count)
        actual_result = ops.reshard_like(replicated, tensor)
        expected_result = tensor
        assert ops.equal(expected_result, actual_result)

    def testReshardLikeShardedToUnsharded(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_dim = 0
        shard_count = 2
        sharded = ops.reshard_split(tensor, dim=shard_dim, count=shard_count)
        actual_result = ops.reshard_like(sharded, tensor)
        expected_result = tensor
        assert ops.equal(expected_result, actual_result)

    def testReshardLikeUnshardedToReplicated(self):
        tensor = torch.rand(4, 5, dtype=torch.float32)
        shard_count = 3
        expected_result = ops.replicate(tensor, count=shard_count)
        actual_result = ops.reshard_like(tensor, expected_result)
        assert expected_result.is_deep_equal(actual_result)

    def testReshardLikeUnshardedToSharded(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_dim = 2
        shard_count = 3
        expected_result = ops.reshard_split(tensor, dim=shard_dim, count=shard_count)
        actual_result = ops.reshard_like(tensor, expected_result)
        assert expected_result.is_deep_equal(actual_result)

    def testReshardLikeShardedToShared(self):
        tensor = torch.rand(5, 6, dtype=torch.float32)
        shard_dim = 1
        shard_count = 3
        expected_result = ops.reshard_split(tensor, dim=shard_dim, count=shard_count)
        actual_result = ops.reshard_like(expected_result, expected_result)
        assert expected_result.is_deep_equal(actual_result)


if __name__ == "__main__":
    unittest.main()
