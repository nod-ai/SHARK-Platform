# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from torch import Tensor, dtype
import torch.nn.functional as F

from ..types import InferenceTensor, ShardedPrimitiveTensor
from ._registry import unbox_tensor
from .signatures import *

# conv2d


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
):
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
        assert False and "Unsupported, TODO: handle sharded channels in input"


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
    assert input.shard_dim == 1 and "Can shard only the channel dimension"
    assert num_groups % input.shard_count == 0 and "Can shard only groups"
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
):
    if lhs.shard_count != rhs.shard_count:
        raise ValueError(
            f"Cannot matmul sharded tensors of different shard_count: "
            f"({lhs.shard_count} vs {rhs.shard_count})"
        )
    partials = [
        matmul(partial_lhs, partial_rhs, transpose_rhs=transpose_rhs)
        for partial_lhs, partial_rhs in zip(lhs.shards, rhs.shards)
    ]
    return ShardedPrimitiveTensor(shard_dim=lhs.shard_dim, ts=partials)


# Sharded sum.


@sharded_cat.override(ShardedPrimitiveTensor)
def sharded_cat_unsharded(maybe_sharded: ShardedPrimitiveTensor):
    shard_ts = [t.as_torch() for t in maybe_sharded.shards]
    return torch.cat(shard_ts, dim=maybe_sharded.shard_dim)


@sharded_sum.override(ShardedPrimitiveTensor)
def sharded_sum_sharded(maybe_sharded: ShardedPrimitiveTensor):
    # TODO: Should implement as an all reduce.
    shards = maybe_sharded.shards
    accum = shards[0].as_torch()
    for shard in shards[1:]:
        accum = torch.add(accum, shard.as_torch())
    return accum
