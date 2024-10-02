# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from torch import Tensor
from typing import List, Optional, Sequence, Union, Any, Tuple
import itertools
from numbers import Number
import math

from ..types import (
    AnyTensor,
    DefaultPrimitiveTensor,
    InferenceTensor,
    PrimitiveTensor,
    ReplicatedTensor,
    ShardedTensor,
    sharding,
    SplitPrimitiveTensor,
    Theta,
    UnreducedTensor,
)
from ..types.tensors import unbox_tensor
from ._registry import AllOfType, AllOfExprsVariadic, IsOfType
from .signatures import *
from .shape import broadcast_dims, broadcast_dim, unbroadcast_dim
from ..utils import longest_equal_range


@all_gather.override(SplitPrimitiveTensor)
def all_gather_split(
    input: SplitPrimitiveTensor, *, dim: int | None
) -> ReplicatedTensor:
    dim = input.shard_dim if dim is None else dim
    # For each device move the shards to it and do a concatenation.
    # If we don't move first, common sub-expression elimination is free to collapse all
    # concatenations into one and then copy to all devices, which is not what we want.
    shards = [
        cat([transfer_to_logical_device(shard, i) for shard in input.shards], dim=dim)
        for i in range(input.shard_count)
    ]
    return ReplicatedTensor(ts=shards)


@all_reduce.override(AllOfType(SplitPrimitiveTensor, UnreducedTensor))
def all_reduce_split_or_unreduced(
    input: Union[SplitPrimitiveTensor, UnreducedTensor],
) -> ReplicatedTensor:
    # For each device move the shards to it and do a reduction.
    # If we don't move first, common sub-expression elimination is free to collapse all
    # reductions into one and then copy to all devices, which is not what we want.
    shards = [
        elementwise(
            torch.add, *[transfer_to_logical_device(shard, i) for shard in input.shards]
        )
        for i in range(input.shard_count)
    ]
    return ReplicatedTensor(ts=shards)


@cat.override(AllOfType(SplitPrimitiveTensor))
def cat_sharded(
    tensors: Sequence[SplitPrimitiveTensor], dim: int
) -> SplitPrimitiveTensor:
    assert len(tensors) > 0
    assert all(
        [
            t.shard_count == tensors[0].shard_count
            and t.shard_dim == tensors[0].shard_dim
            for t in tensors
        ]
    )

    shard_dim = tensors[0].shard_dim
    shard_count = tensors[0].shard_count
    if dim != shard_dim:
        shards = [cat(shards, dim) for shards in zip(*[t.shards for t in tensors])]
        return SplitPrimitiveTensor(ts=shards, shard_dim=shard_dim)
    else:
        # TODO: implement efficient cat along split dim.
        concatenated_unsharded = cat(
            [shard for t in tensors for shard in t.shards], dim
        )
        return reshard_split(concatenated_unsharded, dim=shard_dim, count=shard_count)


# conv2d


def conv2d_all_split(
    input: SplitPrimitiveTensor,
    weight: SplitPrimitiveTensor,
    bias: SplitPrimitiveTensor | None,
    *,
    stride,
    padding,
    dilation,
    groups,
    accum_dtype,
) -> SplitPrimitiveTensor:
    assert accum_dtype is None, "accum_dtype not supported"
    assert input.shard_count == weight.shard_count
    assert bias is None or weight.shard_count == bias.shard_count
    assert (
        input.is_replicated or input.shard_dim == 1
    ), "Only sharding of input channel dimension is supported"
    assert (
        bias is None or weight.shard_dim == 0 and bias.shard_dim == 0
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
    SplitPrimitiveTensor,
    SplitPrimitiveTensor,
    SplitPrimitiveTensor,
    auto_dequant=True,
)(conv2d_all_split)
conv2d.override(SplitPrimitiveTensor, SplitPrimitiveTensor, auto_dequant=True)(
    conv2d_all_split
)


def conv2d_replicated_input_split_weight_and_bias(
    input: ReplicatedTensor,
    weight: SplitPrimitiveTensor,
    bias: SplitPrimitiveTensor | None,
    *,
    stride,
    padding,
    dilation,
    groups,
    accum_dtype,
) -> SplitPrimitiveTensor:
    assert accum_dtype is None, "accum_dtype not supported"
    assert input.shard_count == weight.shard_count
    assert bias is None or weight.shard_count == bias.shard_count
    assert (
        bias is None or weight.shard_dim == 0 and bias.shard_dim == 0
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
    return SplitPrimitiveTensor(shard_dim=1, ts=shards)


conv2d.override(
    ReplicatedTensor, SplitPrimitiveTensor, SplitPrimitiveTensor, auto_dequant=True
)(conv2d_replicated_input_split_weight_and_bias)
conv2d.override(ReplicatedTensor, SplitPrimitiveTensor, auto_dequant=True)(
    conv2d_replicated_input_split_weight_and_bias
)


def conv2d_split_weight_and_bias(
    input: Tensor,
    weight: SplitPrimitiveTensor,
    bias: SplitPrimitiveTensor | None,
    *,
    stride,
    padding,
    dilation,
    groups,
    accum_dtype,
) -> SplitPrimitiveTensor:
    assert accum_dtype is None, "accum_dtype not supported"
    if bias is not None:
        assert weight.shard_count == bias.shard_count

    # Output channels dimension is split.
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
        return SplitPrimitiveTensor(shard_dim=1, ts=shards)
    else:
        assert False, "Unsupported, TODO: handle split channels in input"


conv2d.override(Tensor, SplitPrimitiveTensor, SplitPrimitiveTensor, auto_dequant=True)(
    conv2d_split_weight_and_bias
)
conv2d.override(Tensor, SplitPrimitiveTensor, auto_dequant=True)(
    conv2d_split_weight_and_bias
)

# Sharded elementwise.


@elementwise.override(ReplicatedTensor)
def replicated_elementwise_unary(operator, x: ReplicatedTensor, *args, **kwargs):
    partials = [operator(unbox_tensor(pt), *args, **kwargs) for pt in x.shards]
    return ReplicatedTensor(ts=partials)


@elementwise.override(SplitPrimitiveTensor)
def split_elementwise_unary(operator, x: SplitPrimitiveTensor, *args, **kwargs):
    partials = [operator(unbox_tensor(pt), *args, **kwargs) for pt in x.shards]
    return SplitPrimitiveTensor(shard_dim=x.shard_dim, shape=x.shape, ts=partials)


@elementwise.override(ReplicatedTensor, ReplicatedTensor)
def replicated_elementwise_binary(
    operator, x: ReplicatedTensor, y: ReplicatedTensor, *args, **kwargs
):
    assert x.shard_count == y.shard_count
    shards = [
        operator(unbox_tensor(shard_x), unbox_tensor(shard_y), *args, **kwargs)
        for shard_x, shard_y in zip(x.shards, y.shards)
    ]
    return ReplicatedTensor(ts=shards)


@elementwise.override(SplitPrimitiveTensor, SplitPrimitiveTensor)
def split_elementwise_binary(
    operator, x: SplitPrimitiveTensor, y: SplitPrimitiveTensor, *args, **kwargs
):
    assert x.shard_count == y.shard_count
    x_shard_dim, y_shard_dim = broadcast_dims([x.shard_dim, y.shard_dim], [x, y])
    assert x_shard_dim == y_shard_dim
    pt_xs = [unbox_tensor(pt) for pt in x.shards]
    pt_ys = [unbox_tensor(pt) for pt in y.shards]
    partials = [
        operator(pt_x, pt_y, *args, **kwargs) for pt_x, pt_y in zip(pt_xs, pt_ys)
    ]
    return SplitPrimitiveTensor(
        shard_dim=x.shard_dim,
        shape=torch.broadcast_shapes(x.shape, y.shape),
        ts=partials,
    )


@elementwise.override(SplitPrimitiveTensor, Number)
def elementwise_binary_split_lhs_scalar_rhs(
    operator, x: SplitPrimitiveTensor, y: Number, *args, **kwargs
):
    pt_xs = [unbox_tensor(pt) for pt in x.shards]
    partials = [operator(pt_x, y, *args, **kwargs) for pt_x in pt_xs]
    return SplitPrimitiveTensor(shard_dim=x.shard_dim, shape=x.shape, ts=partials)


@elementwise.override(SplitPrimitiveTensor, Tensor)
def elementwise_binary_split_lhs_tensor_rhs(
    operator, x: SplitPrimitiveTensor, y: Tensor, *args, **kwargs
):
    return elementwise(operator, x, reshard_like(y, like=x), *args, **kwargs)


@elementwise.override(ReplicatedTensor, SplitPrimitiveTensor)
def elementwise_binary_replicated_lhs_sharder_rhs(
    operator, x: ReplicatedTensor, y: SplitPrimitiveTensor, *args, **kwargs
):
    if x.shard_count != y.shard_count:
        raise ValueError(
            f"Operands' number of shards not equal ({x.shard_count} != {y.shard_count})"
        )
    # A replicated tensor can be split with no cost.
    # It is natural to propagate the split instead of the replication.
    x_sharded = reshard_like(x, like=y)
    return elementwise(operator, x_sharded, y, *args, **kwargs)


@elementwise.override(SplitPrimitiveTensor, ReplicatedTensor)
def elementwise_binary_split_lhs_replicated_rhs(
    operator, x: SplitPrimitiveTensor, y: ReplicatedTensor, *args, **kwargs
):
    assert len(y.shape) > 0, "0-rank not supported"
    if x.shard_count != y.shard_count:
        raise ValueError(
            f"Operands' number of shards not equal ({x.shard_count} != {y.shard_count})"
        )

    shard_dim_in_res = broadcast_dim(x.shard_dim, [x.shape, y.shape])
    shard_dim_in_y = unbroadcast_dim(shard_dim_in_res, [y.shape, x.shape])
    is_shard_dim_broadcasted_in_y = (
        shard_dim_in_y is None or y.shape[shard_dim_in_y] == 1
    )
    if is_shard_dim_broadcasted_in_y:
        shards = [
            elementwise(operator, x_shard, y_shard)
            for x_shard, y_shard in zip(x.shards, y.shards)
        ]
        return SplitPrimitiveTensor(ts=shards, shard_dim=shard_dim_in_res)

    y_sharded = reshard_like(y, like=x)
    return elementwise(operator, x, y_sharded, *args, **kwargs)


@elementwise.override(ReplicatedTensor, UnreducedTensor)
def elementwise_binary_replicated_lhs_unreduced_rhs(
    operator, x: ReplicatedTensor, y: UnreducedTensor, *args, **kwargs
):
    if x.shard_count != y.shard_count:
        raise ValueError(
            f"Operands' number of shards not equal ({x.shard_count} != {y.shard_count})"
        )
    y_replicated = reshard_like(y, like=x)
    return elementwise(operator, x, y_replicated, *args, **kwargs)


@elementwise.override(ReplicatedTensor, Tensor)
def elementwise_binary_replicated_lhs_unsharded_rhs(
    operator, x: ReplicatedTensor, y: Tensor, *args, **kwargs
):
    y_replicated = reshard_like(y, like=x)
    return elementwise(operator, x, y_replicated, *args, **kwargs)


@elementwise.override(Tensor, ReplicatedTensor)
def elementwise_binary_replicated_lhs_unsharded_rhs(
    operator, x: Tensor, y: ReplicatedTensor, *args, **kwargs
):
    x_replicated = reshard_like(x, like=y)
    return elementwise(operator, x_replicated, y, *args, **kwargs)


# Embedding Lookup
@embedding_lookup.override(ReplicatedTensor, ReplicatedTensor)
def embedding_lookup_default(
    input: ReplicatedTensor, embedding_matrix: ReplicatedTensor, dtype: torch.dtype
):
    assert input.shard_count == embedding_matrix.shard_count
    shards = [
        embedding_lookup(input_shard, embedding_matrix_shard, dtype)
        for input_shard, embedding_matrix_shard in zip(
            input.shards, embedding_matrix.shards
        )
    ]
    return ReplicatedTensor(ts=shards)


@equal.override(ReplicatedTensor)
def equal_replicated(a: ReplicatedTensor, b: AnyTensor) -> bool:
    return a.is_deep_equal(b)


@equal.override(SplitPrimitiveTensor)
def equal_split(a: SplitPrimitiveTensor, b: AnyTensor) -> bool:
    return a.is_deep_equal(b)


@expand.override(SplitPrimitiveTensor)
def expand_split(
    tensor: SplitPrimitiveTensor, shape: List[int]
) -> SplitPrimitiveTensor:
    assert len(shape) == len(tensor.shape)
    expanded_dims = [
        i
        for i, (old_dim, new_dim) in enumerate(zip(tensor.shape, shape))
        if old_dim == 1 and new_dim != 1
    ]
    assert (
        tensor.shard_dim not in expanded_dims
    ), "Expanding a split dimension is not supported"

    def set_element(l: List, idx: int, el: Any) -> List:
        l[idx] = el
        return l

    shards = [
        expand(
            shard,
            set_element(list(shape), tensor.shard_dim, shard.shape[tensor.shard_dim]),
        )
        for shard in tensor.shards
    ]
    return SplitPrimitiveTensor(ts=shards, shard_dim=tensor.shard_dim)


@flatten.override(ReplicatedTensor)
def flatten_replicated(
    input: ReplicatedTensor, start_dim: int, end_dim: int
) -> ReplicatedTensor:
    shards = [shard.flatten(start_dim, end_dim) for shard in input.shards]
    return ReplicatedTensor(ts=shards)


@flatten.override(SplitPrimitiveTensor)
def flatten_split(
    input: SplitPrimitiveTensor, start_dim: int, end_dim: int
) -> SplitPrimitiveTensor:
    end_dim_resolved = len(input.shape) - 1 if end_dim == -1 else end_dim
    assert input.shard_dim <= start_dim or end_dim_resolved < input.shard_dim, (
        "Flattening of a sharded dimension that is not the leading dimension in the"
        " flattening dimension range is not supported. This would result in a"
        " block-cyclic sharding which is not implemented."
    )
    assert (
        input.shard_dim != start_dim
        or input.shape[input.shard_dim] % input.shard_count == 0
    ), "If the leading flattening dimension is the split dimension, its size must be divisible by the shard count."
    shards = [shard.flatten(start_dim, end_dim) for shard in input.shards]
    shard_dim = (
        input.shard_dim
        if input.shard_dim <= start_dim
        else input.shard_dim - (end_dim_resolved - start_dim)
    )
    return SplitPrimitiveTensor(ts=shards, shard_dim=shard_dim)


@group_norm_affine.override(
    SplitPrimitiveTensor, SplitPrimitiveTensor, SplitPrimitiveTensor
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

    return SplitPrimitiveTensor(shard_dim=1, ts=result_shards)


@index_copy_.override(SplitPrimitiveTensor, ReplicatedTensor, SplitPrimitiveTensor)
def index_copy__split_replicated_split(
    inout: SplitPrimitiveTensor,
    dim: int,
    index: ReplicatedTensor,
    tensor: SplitPrimitiveTensor,
) -> SplitPrimitiveTensor:
    assert (
        inout.shard_count == index.shard_count
        and inout.shard_count == tensor.shard_count
    )
    assert inout.shard_dim == tensor.shard_dim
    assert inout.shard_dim != dim
    for inout_shard, index_shard, tensor_shard in zip(
        inout.shards, index.shards, tensor.shards
    ):
        index_copy_(inout_shard, dim, index_shard, tensor_shard)
    return inout


@index_put_.override(
    AllOfExprsVariadic(
        IsOfType(SplitPrimitiveTensor),
        IsOfType(SplitPrimitiveTensor),
        IsOfType(Tensor, PrimitiveTensor, ReplicatedTensor),
    )
)
def index_put__split(
    inout: SplitPrimitiveTensor,
    indices: Tuple[Union[Tensor, PrimitiveTensor, ReplicatedTensor]],
    values: SplitPrimitiveTensor,
) -> SplitPrimitiveTensor:
    # TODO: verify that the values split dimension is not being indexed or implement
    # this case.
    indices = [replicate(idx, count=inout.shard_count) for idx in indices]
    for i, shard in enumerate(inout.shards):
        shard_indices = [idx.shards[i] for idx in indices]
        shard.index_put_(shard_indices, values.shards[i])
    return inout


@index_select.override(ReplicatedTensor, ReplicatedTensor)
def index_select_replicated(
    tensor: ReplicatedTensor,
    dim: int,
    index: ReplicatedTensor,
) -> ReplicatedTensor:
    assert tensor.shard_count == index.shard_count
    shards = [
        index_select(tensor_shard, dim, index_shard)
        for tensor_shard, index_shard in zip(tensor.shards, index.shards)
    ]
    return ReplicatedTensor(ts=shards)


@index_select.override(SplitPrimitiveTensor, ReplicatedTensor)
def index_select_split_replicated(
    tensor: SplitPrimitiveTensor,
    dim: int,
    index: ReplicatedTensor,
) -> ReplicatedTensor:
    assert tensor.shard_count == index.shard_count
    assert (
        dim != tensor.shard_dim
    ), "Indexing along the split dimension is not supported."
    shards = [
        index_select(tensor_shard, dim, index_shard)
        for tensor_shard, index_shard in zip(tensor.shards, index.shards)
    ]
    return SplitPrimitiveTensor(ts=shards, shard_dim=tensor.shard_dim)


@interpolate.override(ReplicatedTensor)
def interpolate_replicated(
    input: ReplicatedTensor,
    size: Optional[int | List[int]],
    scale_factor: Optional[float | List[float]],
    mode: str,
    align_corners: Optional[bool],
    recompute_scale_factor: Optional[bool],
    antialias: bool,
) -> ReplicatedTensor:
    shards = [
        torch.nn.functional.interpolate(
            input=unbox_tensor(shard),
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )
        for shard in input.shards
    ]
    return ReplicatedTensor(ts=shards)


@interpolate.override(SplitPrimitiveTensor)
def interpolate_split_batch_or_channel(
    input: SplitPrimitiveTensor,
    size: Optional[int | List[int]],
    scale_factor: Optional[float | List[float]],
    mode: str,
    align_corners: Optional[bool],
    recompute_scale_factor: Optional[bool],
    antialias: bool,
) -> SplitPrimitiveTensor:
    assert input.shard_dim == 0 or input.shard_dim == 1
    shards = [
        torch.nn.functional.interpolate(
            input=unbox_tensor(shard),
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )
        for shard in input.shards
    ]
    return SplitPrimitiveTensor(ts=shards, shard_dim=input.shard_dim)


@layer_norm.override(SplitPrimitiveTensor, Tensor, Tensor)
def layer_norm_default(input, weight, bias, *, eps):
    assert input.shard_dim >= 0 and input.shard_dim < len(input.shape) - len(
        weight.shape
    )
    shards = [layer_norm(shard, weight, bias, eps=eps) for shard in input.shards]
    return SplitPrimitiveTensor(shard_dim=input.shard_dim, ts=shards)


# Linear
def linear_sharded(
    input: Tensor | ShardedTensor,
    weight: Tensor | ShardedTensor,
    bias: Tensor | ShardedTensor | None,
    *,
    accum_dtype,
) -> SplitPrimitiveTensor:
    # TODO: handle different dtypes
    result = matmul(input, weight.T)
    if bias is not None:
        result = elementwise(torch.add, result, bias)
    return result


# Override for all cases of Tensor or ShardedTensor arguments,
# except when all Tensors.
# Then we want the default implementation to handle it.
for types in itertools.product([Tensor, ShardedTensor], repeat=3):
    if tuple(types) != (Tensor,) * 3:
        linear.override(*types, auto_dequant=True)(linear_sharded)
for types in itertools.product([Tensor, ShardedTensor], repeat=2):
    if tuple(types) != (Tensor,) * 2:
        linear.override(*types, auto_dequant=True)(linear_sharded)


# Sharded matmuls.


@matmul.override(ReplicatedTensor, SplitPrimitiveTensor)
def matmul_replicated_lhs_split_rhs(
    lhs: ReplicatedTensor, rhs: SplitPrimitiveTensor, *, transpose_rhs: bool
) -> SplitPrimitiveTensor | UnreducedTensor:
    assert lhs.shard_count == rhs.shard_count
    assert len(rhs.shape) == 2

    if transpose_rhs:
        return matmul(lhs, rhs.T)

    rhs_reduction_dim = 1
    if rhs_reduction_dim != rhs.shard_dim:
        lhs_reduction_dimension = len(lhs.shape) - 1
        lhs_split = reshard_split(
            lhs, dim=lhs_reduction_dimension, count=lhs.shard_count
        )
        return matmul(lhs_split, rhs)

    shards = [
        matmul(lhs_shard, rhs_shard)
        for (lhs_shard, rhs_shard) in zip(lhs.shards, rhs.shards)
    ]
    return SplitPrimitiveTensor(ts=shards, shard_dim=len(lhs.shape) - 2 + rhs.shard_dim)


@matmul.override(SplitPrimitiveTensor, Tensor)
def matmul_split_lhs(
    lhs: SplitPrimitiveTensor, rhs, *, transpose_rhs: bool
) -> SplitPrimitiveTensor:
    lhs_reduction_dim = len(lhs.shape) - 1
    assert lhs_reduction_dim != lhs.shard_dim
    shards = [
        matmul(lhs_shard, rhs, transpose_rhs=transpose_rhs) for lhs_shard in lhs.shards
    ]
    return SplitPrimitiveTensor(shard_dim=lhs.shard_dim, ts=shards)


@matmul.override(Tensor, SplitPrimitiveTensor)
def matmul_split_rhs(
    lhs, rhs: SplitPrimitiveTensor, *, transpose_rhs: bool
) -> SplitPrimitiveTensor:
    # When multiplying (unsharded, split), the rhs must be split by column.
    # In a transposed configuration, this is axis 0, otherwise 1.
    # This will result in a ShardedTensor, split by column.
    lhs = unbox_tensor(lhs)
    rhs_shard_dim = rhs.shard_dim
    if transpose_rhs:
        assert (
            rhs_shard_dim == 0
        ), f"matmul[split, transposed rhs] must be split on dim 0 but is {rhs_shard_dim}"
    else:
        assert (
            rhs_shard_dim == 1
        ), f"matmul[split rhs] must be split on dim 1 but is {rhs_shard_dim}"
    partials = [
        matmul(lhs, partial_rhs, transpose_rhs=transpose_rhs)
        for partial_rhs in rhs.shards
    ]
    # The partial is split columnwise (last dim).
    return SplitPrimitiveTensor(shard_dim=len(lhs.shape) - 1, ts=partials)


@matmul.override(SplitPrimitiveTensor, ReplicatedTensor)
def matmul_split_lhs_replicated_rhs(
    lhs: SplitPrimitiveTensor, rhs: ReplicatedTensor, *, transpose_rhs: bool
) -> SplitPrimitiveTensor:
    lhs_reduction_dim = len(lhs.shape) - 1
    assert lhs_reduction_dim != lhs.shard_dim
    if transpose_rhs:
        rhs = rhs.T
    shards = [
        matmul(lhs_shard, rhs_shard)
        for (lhs_shard, rhs_shard) in zip(lhs.shards, rhs.shards)
    ]
    return SplitPrimitiveTensor(ts=shards, shard_dim=lhs.shard_dim)


@matmul.override(SplitPrimitiveTensor, SplitPrimitiveTensor)
def matmul_split(
    lhs: SplitPrimitiveTensor, rhs: SplitPrimitiveTensor, *, transpose_rhs: bool
) -> UnreducedTensor | SplitPrimitiveTensor:
    if lhs.shard_count != rhs.shard_count:
        raise ValueError(
            f"Cannot matmul split tensors of different shard_count: "
            f"({lhs.shard_count} vs {rhs.shard_count})"
        )
    if transpose_rhs:
        return matmul(lhs, rhs.T)

    lhs_reduction_dim = len(lhs.shape) - 1
    rhs_reduction_dim = len(rhs.shape) - 2 if len(rhs.shape) > 1 else len(rhs.shape) - 1

    # The reduction dimension is split on both tensors.
    if lhs_reduction_dim == lhs.shard_dim and rhs_reduction_dim == rhs.shard_dim:
        partials = [
            matmul(partial_lhs, partial_rhs, transpose_rhs=transpose_rhs)
            for partial_lhs, partial_rhs in zip(lhs.shards, rhs.shards)
        ]
        return UnreducedTensor(ts=partials)

    is_batched_matmul = len(lhs.shape) > 2 or len(rhs.shape) > 2
    if (
        is_batched_matmul
        and len(lhs.shape) == len(rhs.shape)
        and lhs.shard_dim == rhs.shard_dim
    ):
        # The same batch dim is sharded for both arguments.
        shards = [
            matmul(lhs_shard, rhs_shard)
            for lhs_shard, rhs_shard in zip(lhs.shards, rhs.shards)
        ]
        return SplitPrimitiveTensor(ts=shards, shard_dim=lhs.shard_dim)

    # -1 for missing parallel dim.
    lhs_parallel_dim = len(lhs.shape) - 2
    rhs_parallel_dim = len(rhs.shape) - 1 if len(rhs.shape) > 1 else -1

    # One parallel dimension is split for each tensor.
    # Or lhs batch dim and rhs parallel dim are split.
    if lhs.shard_dim <= lhs_parallel_dim and rhs_parallel_dim == rhs.shard_dim:
        # We gather along the rhs shard dim.
        # It is more natural to preserve the sharding axis of the input.
        shards = [sharded_cat(matmul(lhs_shard, rhs)) for lhs_shard in lhs.shards]
        return SplitPrimitiveTensor(ts=shards, shard_dim=lhs.shard_dim)

    assert False, "Sharding configuration not supported"


@mean.override(ReplicatedTensor)
def mean_replicated(
    x: ReplicatedTensor,
    dim: Union[int, List[int]],
    keepdim: bool,
    *,
    dtype: torch.dtype,
) -> None:
    shards = [
        torch.mean(unbox_tensor(shard), dim=dim, keepdim=keepdim, dtype=dtype)
        for shard in x.shards
    ]
    return ReplicatedTensor(ts=shards)


@module_register_buffer.override(torch.nn.Module, ShardedTensor)
def module_register_buffer_sharded(
    module: torch.nn.Module, name: str, tensor: ShardedTensor
) -> None:
    for i, shard in enumerate(tensor.shards):
        module_register_buffer(module, f"{name}__shard__{i}", shard)
    setattr(module, name, tensor)


@permute.override(SplitPrimitiveTensor)
def permute_split(tensor: SplitPrimitiveTensor, dims: List[int]):
    permuted_shards = [permute(shard, dims) for shard in tensor.shards]
    permuted_shard_dim = dims[tensor.shard_dim]
    return SplitPrimitiveTensor(ts=permuted_shards, shard_dim=permuted_shard_dim)


@permute.override(ReplicatedTensor)
def permute_replicated(tensor: ReplicatedTensor, dims: List[int]):
    permuted_shards = [permute(shard, dims) for shard in tensor.shards]
    return ReplicatedTensor(ts=permuted_shards)


@replicate.override(ReplicatedTensor)
def replicate_replicated(input: ReplicatedTensor, *, count: int) -> ReplicatedTensor:
    if input.shard_count != count:
        raise ValueError(f"Number of shards not equal ({input.shard_count} != {count})")
    return input


@replicate.override(UnreducedTensor)
def replicate_unreduced(input: UnreducedTensor, *, count: int) -> ReplicatedTensor:
    if input.shard_count != count:
        raise ValueError(f"Number of shards not equal ({input.shard_count} != {count})")
    return all_reduce(input)


@replicate.override(Tensor)
def replicate_unsharded(input, *, count: int) -> ReplicatedTensor:
    torch_input = unbox_tensor(input)
    return ReplicatedTensor(ts=torch_input, shard_count=count)


@reshape.override(SplitPrimitiveTensor)
def reshape_split(
    tensor: SplitPrimitiveTensor, shape: List[int]
) -> SplitPrimitiveTensor:
    if _reshape_get_single_split_dim(tensor.shape, shape) is not None:
        return view(tensor, shape)

    flatten_dim_range = _reshape_get_flatten_dim_range(tensor.shape, shape)
    if flatten_dim_range is not None:
        return flatten(tensor, flatten_dim_range[0], flatten_dim_range[1] - 1)

    raise ValueError(
        f"Unsupported reshaping of sharded split tensor of shape {tensor.shape} to shape {shape}"
    )


@reshard.override(Tensor, sharding.Split)
def reshard_tensor_split(input: Tensor, spec: sharding.Split) -> AnyTensor:
    return reshard_split(input, dim=spec.shard_dim, count=spec.shard_count)


@reshard.override(Theta, sharding.ThetaLayerSharding)
def reshard_theta_layer_sharding(
    input: Theta, spec: sharding.ThetaLayerSharding
) -> Theta:
    return reshard(input, spec.theta_sharding())


@reshard.override(Theta, sharding.ThetaSharding)
def reshard_theta_sharding(input: Theta, spec: sharding.ThetaSharding) -> Theta:
    def make_value(input: Theta | InferenceTensor, spec) -> dict | InferenceTensor:
        result = reshard(input, spec)
        if isinstance(result, Theta):
            result = result.tree
        elif isinstance(result, torch.Tensor):
            result = DefaultPrimitiveTensor(data=result, name=input.name)
        else:
            assert isinstance(result, InferenceTensor)
            result.name = input.name
        return result

    return Theta(
        {
            k: make_value(input(k), spec[k])
            for k in input.keys
            if not isinstance(spec[k], sharding.Ignore)
        }
    )


@reshard.override(Theta, sharding.ThetaLayerSharding)
def reshard_theta_layer_sharding(
    input: Theta, spec: sharding.ThetaLayerSharding
) -> Theta:
    return reshard(input, spec.theta_sharding())


@reshard.override(object, sharding.Unsharded)
def reshard_all_to_unsharded(input: AnyTensor, spec: sharding.Unsharded) -> Tensor:
    return unshard(input)


@reshard.override(object, sharding.Replicated)
def reshard_all_to_replicated(
    input: AnyTensor, spec: sharding.Replicated
) -> ReplicatedTensor:
    return replicate(input, spec.shard_count)


@reshard_split.override(Tensor)
def reshard_split_unsharded(input, *, dim: int, count: int) -> SplitPrimitiveTensor:
    torch_input = unbox_tensor(input)
    return SplitPrimitiveTensor(ts=torch_input, shard_dim=dim, shard_count=count)


@reshard_split.override(SplitPrimitiveTensor)
def reshard_split_split(
    input: SplitPrimitiveTensor, *, dim: int, count: int
) -> SplitPrimitiveTensor:
    if input.shard_count != count:
        raise ValueError(f"Number of shards not equal ({input.shard_count} != {count})")
    if input.shard_dim != dim:
        raise ValueError(f"Resharding is not supported")
    return input


@reshard_split.override(ReplicatedTensor)
def reshard_split_replicated(
    input: ReplicatedTensor, *, dim: int, count: int
) -> SplitPrimitiveTensor:
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
    return SplitPrimitiveTensor(ts=shards, shard_dim=dim)


@reshard_like.override(Tensor, SplitPrimitiveTensor)
def reshard_like_unsharded_to_split(
    input, like: SplitPrimitiveTensor
) -> SplitPrimitiveTensor:
    torch_input = unbox_tensor(input)
    return reshard_split(torch_input, dim=like.shard_dim, count=like.shard_count)


@reshard_like.override(ReplicatedTensor, Tensor)
def reshard_like_replicated_to_unsharded(input: ReplicatedTensor, like):
    return input.shards[0]


@reshard_like.override(SplitPrimitiveTensor, Tensor)
def reshard_like_split_to_unsharded(input: SplitPrimitiveTensor, like):
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


@reshard_like.override(ReplicatedTensor, SplitPrimitiveTensor)
def reshard_like_replicated_to_split(
    tensor: ReplicatedTensor, like: SplitPrimitiveTensor
) -> SplitPrimitiveTensor:
    return reshard_split(tensor, dim=like.shard_dim, count=like.shard_count)


@reshard_like.override(SplitPrimitiveTensor, SplitPrimitiveTensor)
def reshard_like_split_to_split(
    tensor: SplitPrimitiveTensor, like: SplitPrimitiveTensor
) -> SplitPrimitiveTensor:
    assert (
        tensor.shard_count == like.shard_count and tensor.shard_dim == like.shard_dim
    ), "Resharding is not supported"
    return tensor


@reshard_like.override(UnreducedTensor, ReplicatedTensor)
def reshard_like_unreduced_to_replicated(
    tensor: UnreducedTensor, like: ReplicatedTensor
) -> ReplicatedTensor:
    return replicate(tensor, count=like.shard_count)


@sharded_cat.override(SplitPrimitiveTensor)
def sharded_cat_unsharded(maybe_sharded: SplitPrimitiveTensor):
    shard_ts = [t.as_torch() for t in maybe_sharded.shards]
    return torch.cat(shard_ts, dim=maybe_sharded.shard_dim)


# Sharded sum.


def _sharded_sum_sharded(tensor: ShardedTensor) -> Tensor:
    accum = tensor.shards[0].as_torch()
    for shard in tensor.shards[1:]:
        accum = torch.add(accum, shard.as_torch())
    return accum


@sharded_sum.override(SplitPrimitiveTensor)
def sharded_sum_split(maybe_sharded: SplitPrimitiveTensor) -> Tensor:
    # TODO: Should implement as an all reduce.
    return _sharded_sum_sharded(maybe_sharded)


@sharded_sum.override(UnreducedTensor)
def sharded_sum_unreduced(maybe_sharded: UnreducedTensor) -> Tensor:
    return _sharded_sum_sharded(maybe_sharded)


@softmax.override(SplitPrimitiveTensor)
def softmax_split(
    tensor: SplitPrimitiveTensor, dim: Optional[int], dtype: Optional[torch.dtype]
) -> Tensor:
    dim = dim if dim is None or dim >= 0 else len(tensor.shape) + dim
    assert (
        dim is not None and dim != tensor.shard_dim
    ), "Softmax along split dimension is not supported."
    shards = [softmax(shard, dim=dim, dtype=dtype) for shard in tensor.shards]
    return SplitPrimitiveTensor(
        ts=shards, shard_dim=tensor.shard_dim, shape=tensor.shape
    )


@to.override(ReplicatedTensor)
def to_replicated(tensor: ReplicatedTensor, *args, **kwargs):
    shards = [to(shard, *args, **kwargs) for shard in tensor.shards]
    return ReplicatedTensor(ts=shards)


@to.override(SplitPrimitiveTensor)
def to_split(tensor: SplitPrimitiveTensor, *args, **kwargs):
    shards = [to(shard, *args, **kwargs) for shard in tensor.shards]
    return SplitPrimitiveTensor(ts=shards, shard_dim=tensor.shard_dim)


@transpose.override(SplitPrimitiveTensor)
def transpose_split(
    tensor: SplitPrimitiveTensor, dim0: int, dim1: int
) -> SplitPrimitiveTensor:
    shards = [transpose(shard, dim0, dim1) for shard in tensor.shards]
    shard_dim = tensor.shard_dim
    if shard_dim == dim0:
        shard_dim = dim1
    elif shard_dim == dim1:
        shard_dim = dim0
    return SplitPrimitiveTensor(ts=shards, shard_dim=shard_dim)


@unflatten.override(SplitPrimitiveTensor)
def unflatten_split(
    input: SplitPrimitiveTensor, dim: int, sizes: Tuple[int]
) -> SplitPrimitiveTensor:
    assert dim != input.shard_dim, "Unflattening the split dimension is not supported."
    shards = [unflatten(shard, dim, sizes) for shard in input.shards]
    shard_dim = input.shard_dim
    if dim < shard_dim:
        shard_dim += len(sizes) - 1
    return SplitPrimitiveTensor(ts=shards, shard_dim=shard_dim)


@unshard.override(ReplicatedTensor)
def unshard_replicated(input: ReplicatedTensor) -> Tensor:
    return input.shards[0]


@unshard.override(SplitPrimitiveTensor)
def unshard_split(input: SplitPrimitiveTensor) -> Tensor:
    return sharded_cat(input)


@unshard.override(UnreducedTensor)
def unshard_unreduced(input: UnreducedTensor) -> Tensor:
    return elementwise(torch.add, *input.shards)


@unshard.override(Tensor)
def unshard_unsharded(input: Tensor) -> Tensor:
    return input


def _reshape_get_flatten_dim_range(
    from_shape: List[int], to_shape: List[int]
) -> Optional[Tuple[int, int]]:
    """If a reshape would flatten a range of dimensions return that index range [begin, end).
    If the reshape is not of that kind return `None`."""
    flatten_start_len = _reshape_get_single_split_dim(to_shape, from_shape)
    if flatten_start_len is None:
        return None
    start, length = flatten_start_len
    return start, start + length


def _reshape_infer_dynamic_dim(
    shape1: List[int], shape2: List[int]
) -> Tuple[List[int], List[int]]:
    assert (
        len([d for d in list(shape1) + list(shape2) if d < 0]) <= 1
    ), "Only one dynamic dimension is allowed"
    shape1_dynamic_dims = [i for i, d in enumerate(shape1) if d <= 0]
    if len(shape1_dynamic_dims) > 0:
        s2, s1 = _reshape_infer_dynamic_dim(shape2, shape1)
        return s1, s2

    shape2_dynamic_dims = [i for i, d in enumerate(shape2) if d <= 0]
    if len(shape2_dynamic_dims) == 0:
        return shape1, shape2
    shape2_dynamic_dim = shape2_dynamic_dims[0]
    shape1_size = math.prod(shape1)
    shape2_size_without_dynamic_dim = math.prod(d for d in shape2 if d > 0)
    shape2_res = list(shape2)
    assert shape1_size % shape2_size_without_dynamic_dim == 0
    shape2_res[shape2_dynamic_dim] = shape1_size // shape2_size_without_dynamic_dim
    assert shape2_res[shape2_dynamic_dim] > 0
    return shape1, shape2_res


def _reshape_get_single_split_dim(
    from_shape: List[int], to_shape: List[int]
) -> Optional[Tuple[int, int]]:
    """If a reshape would split a single dimension, return its index and the length of the new dimensions.
    If the reshape is not of that kind return `None`."""
    from_shape, to_shape = _reshape_infer_dynamic_dim(from_shape, to_shape)

    if len(to_shape) < len(from_shape):
        return None
    i = longest_equal_range(from_shape, to_shape)
    if i == len(from_shape):
        return i
    j = len(to_shape) - longest_equal_range(reversed(from_shape), reversed(to_shape))
    assert i < j
    expected_split_dim_size = math.prod(to_shape[i:j])
    if expected_split_dim_size != from_shape[i]:
        return None
    return (
        i,
        j - i,
    )


@unsqueeze.override(SplitPrimitiveTensor)
def unsqueeze_split(tensor: SplitPrimitiveTensor, dim: int) -> SplitPrimitiveTensor:
    shards = [torch.unsqueeze(unbox_tensor(shard), dim) for shard in tensor.shards]
    shard_dim = tensor.shard_dim
    dim_resolved = dim if dim >= 0 else dim + len(tensor.shape) + 1
    if shard_dim >= dim_resolved:
        shard_dim += 1
    return SplitPrimitiveTensor(ts=shards, shard_dim=shard_dim)


@unsqueeze.override(ReplicatedTensor)
def unsqueeze_replicated(tensor: ReplicatedTensor, dim: int) -> SplitPrimitiveTensor:
    shards = [torch.unsqueeze(unbox_tensor(shard), dim) for shard in tensor.shards]
    return ReplicatedTensor(ts=shards)


@view.override(SplitPrimitiveTensor)
def view_split(tensor: SplitPrimitiveTensor, shape: List[int]) -> SplitPrimitiveTensor:
    view_split_range = _reshape_get_single_split_dim(tensor.shape, shape)
    if view_split_range is None:
        raise ValueError(
            "Only taking a tensor view where splitting a single dimension is supported"
        )
    view_split_dim = view_split_range[0]

    if view_split_dim == tensor.shard_dim:
        if tensor.shape[view_split_dim] % tensor.shard_count != 0:
            raise ValueError(
                "Only splitting a dimension that is multiple of the shard count is supported"
            )
        if shape[view_split_dim] % tensor.shard_count != 0:
            raise ValueError(
                "The resulting leading splitting dimension must be multiple of the shard count"
            )

    shard_dim = tensor.shard_dim
    if shard_dim > view_split_dim:
        new_dims_count = len(shape) - len(tensor.shape)
        shard_dim += new_dims_count
    new_shard_shape = list(shape)
    new_shard_shape[shard_dim] //= tensor.shard_count
    shards = [view(shard, new_shard_shape) for shard in tensor.shards]
    res = SplitPrimitiveTensor(shard_dim=shard_dim, ts=shards)
    assert math.prod(res.shape) == math.prod(tensor.shape)
    return res
