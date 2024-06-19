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
from ._registry import unbox_tensor
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
        # TODO: handle "+"
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


@matmul.override(Tensor, ShardedPrimitiveTensor)
def matmul_sharded_rhs(lhs, rhs: ShardedPrimitiveTensor, *, transpose_rhs: bool):
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


@matmul.override(ShardedPrimitiveTensor, ShardedPrimitiveTensor)
def matmul_sharded(
    lhs: ShardedPrimitiveTensor, rhs: ShardedPrimitiveTensor, *, transpose_rhs: bool
) -> UnreducedTensor:
    lhs_reduction_dim = len(lhs.shape) - 1
    rhs_reduction_dim = 1 if transpose_rhs else 0
    assert (
        lhs_reduction_dim == lhs.shard_dim and rhs_reduction_dim == rhs.shard_dim
    ), "Only sharding of the reduction dimension is supported"

    if lhs.shard_count != rhs.shard_count:
        raise ValueError(
            f"Cannot matmul sharded tensors of different shard_count: "
            f"({lhs.shard_count} vs {rhs.shard_count})"
        )
    partials = [
        matmul(partial_lhs, partial_rhs, transpose_rhs=transpose_rhs)
        for partial_lhs, partial_rhs in zip(lhs.shards, rhs.shards)
    ]
    return UnreducedTensor(ts=partials)


@permute.override(ShardedPrimitiveTensor)
def permute_sharded(tensor: ShardedPrimitiveTensor, dims: List[int]):
    permuted_shards = [permute(shard, dims) for shard in tensor.shards]
    permuted_shard_dim = dims[tensor.shard_dim]
    return ShardedPrimitiveTensor(ts=permuted_shards, shard_dim=permuted_shard_dim)


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
