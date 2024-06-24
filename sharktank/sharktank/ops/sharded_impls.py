# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from torch import Tensor
from typing import List

from ..types import (
    ShardedPrimitiveTensor,
    ReplicatedTensor,
    UnreducedTensor,
    ShardedTensor,
)
from ._registry import unbox_tensor, AnyTensor
from .signatures import *


@all_gather.override(ShardedPrimitiveTensor)
def all_gather_sharded(
    input: ShardedPrimitiveTensor, *, dim: int | None
) -> ReplicatedTensor:
    assert (
        dim is None
    ), "gather dimension other than `input.shard_dim` is not supported."
    # TODO: figure out how to avoid common sub-expression elimination to not
    # merge all these into one.
    # Even if we place each resulting shard inside of ReplicatedTensor on a
    # distinct logical device with an explicit operation, CSE should still
    # collapse them.
    shards = [sharded_cat(input) for i in range(input.shard_count)]
    return ReplicatedTensor(ts=shards)


# conv2d


def conv2d_all_sharded(
    input: ShardedPrimitiveTensor,
    weight: ShardedPrimitiveTensor,
    bias: ShardedPrimitiveTensor | None,
    *,
    stride,
    padding,
    dilation,
    groups,
) -> ShardedPrimitiveTensor:
    assert input.shard_count == weight.shard_count
    assert bias is None or weight.shard_count == bias.shard_count
    assert (
        input.is_replicated or input.shard_dim == 1
    ), "Only sharding of input channel dimension is supported"
    assert (
        weight.shard_dim == 0 and bias.shard_dim == 0
    ), "Only sharding of output channel dimension is supported"

    # TODO: allow for implementation where we don't all-gather, but gather
    # instead and share the input tensor.
    # This may be useful when having peered memory.
    #
    # Another option is to have each device do multiple convolutions without
    # doing an gather/all-gather.
    # Then a reduction across the shards.
    # If groups are divisible by the number of shards we don't need to do a
    # reduction.
    # We would be relaying on the compiler to fuse the convs into a single
    # kernel.
    # A batched conv where the mini-batches(shards) are scattered across
    # multiple buffers.
    #
    # With tuning allow for selection of the appropriate version.

    input = all_gather(input)

    return conv2d(
        input,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


conv2d.override(
    ShardedPrimitiveTensor,
    ShardedPrimitiveTensor,
    ShardedPrimitiveTensor,
    auto_dequant=True,
)(conv2d_all_sharded)
conv2d.override(ShardedPrimitiveTensor, ShardedPrimitiveTensor, auto_dequant=True)(
    conv2d_all_sharded
)


def conv2d_replicated_input_sharded_weight_and_bias(
    input: ReplicatedTensor,
    weight: ShardedPrimitiveTensor,
    bias: ShardedPrimitiveTensor | None,
    *,
    stride,
    padding,
    dilation,
    groups,
) -> ShardedPrimitiveTensor:
    assert input.shard_count == weight.shard_count
    assert bias is None or weight.shard_count == bias.shard_count
    assert (
        weight.shard_dim == 0 and bias.shard_dim == 0
    ), "Only sharding of output channel dimension is supported"
    assert groups == 1

    shards = [
        conv2d(
            x,
            w,
            b,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        for x, w, b in zip(
            input.shards,
            weight.shards,
            [None] * weight.shard_count if bias is None else bias.shards,
        )
    ]
    return ShardedPrimitiveTensor(shard_dim=1, ts=shards)


conv2d.override(
    ReplicatedTensor, ShardedPrimitiveTensor, ShardedPrimitiveTensor, auto_dequant=True
)(conv2d_all_sharded)
conv2d.override(ReplicatedTensor, ShardedPrimitiveTensor, auto_dequant=True)(
    conv2d_all_sharded
)


def conv2d_sharded_weight_and_bias(
    input: Tensor,
    weight: ShardedPrimitiveTensor,
    bias: ShardedPrimitiveTensor | None,
    *,
    stride,
    padding,
    dilation,
    groups,
    accum_dtype,
) -> ShardedPrimitiveTensor:
    assert weight.shard_count == bias.shard_count

    # Output channels dimension is sharded.
    if weight.shard_dim == 0 and groups == 1:
        assert bias is None or bias.shard_dim == 0
        shards = [
            conv2d(
                input,
                w,
                b,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            for w, b in zip(
                weight.shards,
                [None] * weight.shard_count if bias is None else bias.shards,
            )
        ]
        return ShardedPrimitiveTensor(shard_dim=1, ts=shards)
    else:
        assert False, "Unsupported, TODO: handle sharded channels in input"


conv2d.override(
    Tensor, ShardedPrimitiveTensor, ShardedPrimitiveTensor, auto_dequant=True
)(conv2d_sharded_weight_and_bias)
conv2d.override(Tensor, ShardedPrimitiveTensor, auto_dequant=True)(
    conv2d_sharded_weight_and_bias
)

# Sharded elementwise.


@elementwise.override(ShardedPrimitiveTensor)
def sharded_elementwise_unary(operator, x: ShardedPrimitiveTensor):
    partials = [operator(unbox_tensor(pt)) for pt in x.shards]
    return ShardedPrimitiveTensor(shard_dim=x.shard_dim, shape=x.shape, ts=partials)


@elementwise.override(ShardedPrimitiveTensor, ShardedPrimitiveTensor)
def sharded_elementwise_binary(
    operator, x: ShardedPrimitiveTensor, y: ShardedPrimitiveTensor
):
    assert x.shard_count == y.shard_count
    assert x.shard_dim == y.shard_dim
    pt_xs = [unbox_tensor(pt) for pt in x.shards]
    pt_ys = [unbox_tensor(pt) for pt in y.shards]
    partials = [operator(pt_x, pt_y) for pt_x, pt_y in zip(pt_xs, pt_ys)]
    return ShardedPrimitiveTensor(shard_dim=x.shard_dim, shape=x.shape, ts=partials)


@elementwise.override(ReplicatedTensor, ShardedPrimitiveTensor)
def elementwise_binary_replicated_lhs_sharder_rhs(
    operator, x: ReplicatedTensor, y: ShardedPrimitiveTensor
):
    if x.shard_count != y.shard_count:
        raise ValueError(
            f"Operands' number of shards not equal ({x.shard_count} != {y.shard_count})"
        )
    # A replicated tensor can be split with no cost.
    # It is natural to propagate the split instead of the replication.
    x_sharded = reshard_like(x, like=y)
    return elementwise(operator, x_sharded, y)


@elementwise.override(ShardedPrimitiveTensor, ReplicatedTensor)
def elementwise_binary_sharded_lhs_replicated_rhs(
    operator, x: ReplicatedTensor, y: ShardedPrimitiveTensor
):
    if x.shard_count != y.shard_count:
        raise ValueError(
            f"Operands' number of shards not equal ({x.shard_count} != {y.shard_count})"
        )
    y_sharded = reshard_like(y, like=x)
    return elementwise(operator, x, y_sharded)


@equal.override(ReplicatedTensor)
def equal_replicated(a: ReplicatedTensor, b: AnyTensor) -> bool:
    return a.is_deep_equal(b)


@equal.override(ShardedPrimitiveTensor)
def equal_sharded(a: ShardedPrimitiveTensor, b: AnyTensor) -> bool:
    return a.is_deep_equal(b)


@group_norm_affine.override(
    ShardedPrimitiveTensor, ShardedPrimitiveTensor, ShardedPrimitiveTensor
)
def shareded_group_norm_affine(input, weight, bias, *, num_groups, eps):
    assert (
        input.shard_count == weight.shard_count
        and input.shard_count == bias.shard_count
    )
    assert input.shard_dim == 1, "Can shard only the channel dimension"
    assert num_groups % input.shard_count == 0, "Can shard only groups"
    num_groups_per_shard = num_groups // input.shard_count

    result_shards = [
        group_norm_affine(x, num_groups=num_groups_per_shard, weight=w, bias=b, eps=eps)
        for x, w, b in zip(input.shards, weight.shards, bias.shards)
    ]

    return ShardedPrimitiveTensor(shard_dim=1, ts=result_shards)


@layer_norm.override(ShardedPrimitiveTensor, Tensor, Tensor)
def layer_norm_default(input, weight, bias, *, eps):
    assert input.shard_dim >= 0 and input.shard_dim < len(input.shape) - len(
        weight.shape
    )
    shards = [layer_norm(shard, weight, bias, eps=eps) for shard in input.shards]
    return ShardedPrimitiveTensor(shard_dim=input.shard_dim, ts=shards)


# Linear
def linear_sharded(
    input: Tensor | ShardedPrimitiveTensor,
    weight: ShardedPrimitiveTensor,
    bias: ShardedPrimitiveTensor | None,
    *,
    accum_dtype,
) -> Tensor | ShardedPrimitiveTensor:
    # TODO: handle different dtypes
    result = matmul(input, weight.T)
    if bias is not None:
        result = result + bias
    return result


linear.override(Tensor, ShardedPrimitiveTensor, auto_dequant=True)(linear_sharded)
linear.override(
    Tensor, ShardedPrimitiveTensor, ShardedPrimitiveTensor, auto_dequant=True
)(linear_sharded)
linear.override(ShardedPrimitiveTensor, ShardedPrimitiveTensor, auto_dequant=True)(
    linear_sharded
)
linear.override(
    ShardedPrimitiveTensor,
    ShardedPrimitiveTensor,
    ShardedPrimitiveTensor,
    auto_dequant=True,
)(linear_sharded)


# Sharded matmuls.


@matmul.override(ShardedPrimitiveTensor, Tensor)
def matmul_sharded_lhs(
    lhs: ShardedPrimitiveTensor, rhs, *, transpose_rhs: bool
) -> ShardedPrimitiveTensor:
    lhs_reduction_dim = len(lhs.shape) - 1
    assert lhs_reduction_dim != lhs.shard_dim
    shards = [
        matmul(lhs_shard, rhs, transpose_rhs=transpose_rhs) for lhs_shard in lhs.shards
    ]
    return ShardedPrimitiveTensor(shard_dim=lhs.shard_dim, ts=shards)


@matmul.override(Tensor, ShardedPrimitiveTensor)
def matmul_sharded_rhs(
    lhs, rhs: ShardedPrimitiveTensor, *, transpose_rhs: bool
) -> ShardedPrimitiveTensor:
    # When multiplying (unsharded, sharded), the rhs must be sharded by column.
    # In a transposed configuration, this is axis 0, otherwise 1.
    # This will result in a ShardedTensor, sharded by column.
    lhs = unbox_tensor(lhs)
    rhs_shard_dim = rhs.shard_dim
    if transpose_rhs:
        assert (
            rhs_shard_dim == 0
        ), f"matmul[sharded, transposed rhs] must be sharded on dim 0 but is {rhs_shard_dim}"
    else:
        assert (
            rhs_shard_dim == 1
        ), f"matmul[sharded rhs] must be sharded on dim 1 but is {rhs_shard_dim}"
    partials = [
        matmul(lhs, partial_rhs, transpose_rhs=transpose_rhs)
        for partial_rhs in rhs.shards
    ]
    # The partial is sharded columnwise (last dim).
    return ShardedPrimitiveTensor(shard_dim=len(lhs.shape) - 1, ts=partials)


@matmul.override(ShardedPrimitiveTensor, ReplicatedTensor)
def matmul_sharded_lhs_replicated_rhs(
    lhs: ShardedPrimitiveTensor, rhs: ReplicatedTensor, *, transpose_rhs: bool
) -> ShardedPrimitiveTensor:
    lhs_reduction_dim = len(lhs.shape) - 1
    assert lhs_reduction_dim != lhs.shard_dim
    if transpose_rhs:
        rhs = rhs.T
    shards = [
        matmul(lhs_shard, rhs_shard)
        for (lhs_shard, rhs_shard) in zip(lhs.shards, rhs.shards)
    ]
    return ShardedPrimitiveTensor(ts=shards, shard_dim=lhs.shard_dim)


@matmul.override(ShardedPrimitiveTensor, ShardedPrimitiveTensor)
def matmul_sharded(
    lhs: ShardedPrimitiveTensor, rhs: ShardedPrimitiveTensor, *, transpose_rhs: bool
) -> UnreducedTensor | ShardedPrimitiveTensor:
    if lhs.shard_count != rhs.shard_count:
        raise ValueError(
            f"Cannot matmul sharded tensors of different shard_count: "
            f"({lhs.shard_count} vs {rhs.shard_count})"
        )

    lhs_reduction_dim = len(lhs.shape) - 1
    rhs_reduction_dim = 1 if transpose_rhs else 0

    # The reduction dimension is sharded on both tensors.
    if lhs_reduction_dim == lhs.shard_dim and rhs_reduction_dim == rhs.shard_dim:
        partials = [
            matmul(partial_lhs, partial_rhs, transpose_rhs=transpose_rhs)
            for partial_lhs, partial_rhs in zip(lhs.shards, rhs.shards)
        ]
        return UnreducedTensor(ts=partials)

    # One parallel dimension is sharded for each tensor.
    if lhs_reduction_dim != lhs.shard_dim and rhs_reduction_dim != rhs.shard_dim:
        if transpose_rhs:
            rhs = rhs.T
        # We gather along the rhs shard dim.
        # It is more natural to preserve the sharding axis of the input.
        shards = [sharded_cat(matmul(lhs_shard, rhs)) for lhs_shard in lhs.shards]
        return ShardedPrimitiveTensor(ts=shards, shard_dim=lhs.shard_dim)

    assert False, "Sharding configuration not supported"


@permute.override(ShardedPrimitiveTensor)
def permute_sharded(tensor: ShardedPrimitiveTensor, dims: List[int]):
    permuted_shards = [permute(shard, dims) for shard in tensor.shards]
    permuted_shard_dim = dims[tensor.shard_dim]
    return ShardedPrimitiveTensor(ts=permuted_shards, shard_dim=permuted_shard_dim)


@replicate.override(ReplicatedTensor)
def replicate_replicated(input: ReplicatedTensor, *, count: int) -> ReplicatedTensor:
    if input.shard_count != count:
        raise ValueError(f"Number of shards not equal ({input.shard_count} != {count})")
    assert input.shard_count == count
    return input


@replicate.override(Tensor)
def replicate_unsharded(input, *, count: int) -> ReplicatedTensor:
    torch_input = unbox_tensor(input)
    return ReplicatedTensor(ts=torch_input, shard_count=count)


@reshard.override(Tensor)
def shard_unsharded(input, *, dim: int, count: int) -> ShardedPrimitiveTensor:
    torch_input = unbox_tensor(input)
    return ShardedPrimitiveTensor(ts=torch_input, shard_dim=dim, shard_count=count)


@reshard.override(ShardedPrimitiveTensor)
def shard_sharded(
    input: ShardedPrimitiveTensor, *, dim: int, count: int
) -> ShardedPrimitiveTensor:
    if input.shard_count != count:
        raise ValueError(f"Number of shards not equal ({input.shard_count} != {count})")
    if input.shard_dim != dim:
        raise ValueError(f"Resharding is not supported")
    return input


@reshard.override(ReplicatedTensor)
def shard_replicated(
    input: ReplicatedTensor, *, dim: int, count: int
) -> ShardedPrimitiveTensor:
    if input.shard_count != count:
        raise ValueError(f"Number of shards not equal ({input.shard_count} != {count})")

    def slice_range_along_dim(dim: int, start: int, end: int):
        res = [slice(None)] * len(input.shape)
        res[dim] = slice(start, end)
        return res

    shard_size_along_dim = input.shape[dim] // count
    shards = [
        unbox_tensor(shard)[
            slice_range_along_dim(
                dim=dim,
                start=shard_idx * shard_size_along_dim,
                end=(shard_idx + 1) * shard_size_along_dim,
            )
        ]
        for shard_idx, shard in enumerate(input.shards)
    ]
    return ShardedPrimitiveTensor(ts=shards, shard_dim=dim)


@reshard_like.override(Tensor, ShardedPrimitiveTensor)
def reshard_like_unsharded_to_sharded(
    input, like: ShardedPrimitiveTensor
) -> ShardedPrimitiveTensor:
    torch_input = unbox_tensor(input)
    return reshard(torch_input, dim=like.shard_dim, count=like.shard_count)


@reshard_like.override(ReplicatedTensor, Tensor)
def reshard_like_replicated_to_unsharded(input: ReplicatedTensor, like):
    return input.shards[0]


@reshard_like.override(ShardedPrimitiveTensor, Tensor)
def reshard_like_sharded_to_unsharded(input: ShardedPrimitiveTensor, like):
    return sharded_cat(input)


@reshard_like.override(Tensor, ReplicatedTensor)
def reshard_like_unsharded_to_replicated(
    tensor, like: ReplicatedTensor
) -> ReplicatedTensor:
    torch_tensor = unbox_tensor(tensor)
    return replicate(torch_tensor, count=like.shard_count)


@reshard_like.override(ReplicatedTensor, ReplicatedTensor)
def reshard_like_replicated_to_replicated(
    tensor: ReplicatedTensor, like: ReplicatedTensor
) -> ReplicatedTensor:
    if tensor.shard_count != like.shard_count:
        raise ValueError(
            f"Operands' number of shards not equal ({input.shard_count} != {like.shard_count})"
        )
    return tensor


@reshard_like.override(ReplicatedTensor, ShardedPrimitiveTensor)
def reshard_like_replicated_to_sharded(
    tensor: ReplicatedTensor, like: ShardedPrimitiveTensor
) -> ShardedPrimitiveTensor:
    return reshard(tensor, dim=like.shard_dim, count=like.shard_count)


@reshard_like.override(ShardedPrimitiveTensor, ShardedPrimitiveTensor)
def reshard_like_sharded_to_sharded(
    tensor: ShardedPrimitiveTensor, like: ShardedPrimitiveTensor
) -> ShardedPrimitiveTensor:
    assert (
        tensor.shard_count == like.shard_count and tensor.shard_dim == like.shard_dim
    ), "Resharding is not supported"
    return tensor


# Sharded sum.


@sharded_cat.override(ShardedPrimitiveTensor)
def sharded_cat_unsharded(maybe_sharded: ShardedPrimitiveTensor):
    shard_ts = [t.as_torch() for t in maybe_sharded.shards]
    return torch.cat(shard_ts, dim=maybe_sharded.shard_dim)


def _sharded_sum_sharded(tensor: ShardedTensor) -> Tensor:
    accum = tensor.shards[0].as_torch()
    for shard in tensor.shards[1:]:
        accum = torch.add(accum, shard.as_torch())
    return accum


@sharded_sum.override(ShardedPrimitiveTensor)
def sharded_sum_sharded(maybe_sharded: ShardedPrimitiveTensor):
    # TODO: Should implement as an all reduce.
    return _sharded_sum_sharded(maybe_sharded)


@sharded_sum.override(UnreducedTensor)
def sharded_sum_sharded(maybe_sharded: UnreducedTensor):
    return _sharded_sum_sharded(maybe_sharded)
