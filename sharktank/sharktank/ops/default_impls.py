# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file contains overrides of the standard ops for normal torch and
# generic primitive/quantized types.

from typing import Optional, List, Sequence, Union, Tuple

import torch
from torch import Tensor, dtype
import torch.nn.functional as F
from numbers import Number

from ..types import (
    PrimitiveTensor,
    QuantizedTensor,
    InferenceTensor,
    PlanarQuantizedTensor,
    BlockScaledI4Layout,
)
from ..types.tensors import unbox_tensor, AnyTensor
from ._registry import AllOfType, AllOfExprs, AllOfExprsVariadic, IsOfType
from .signatures import *
import iree.turbine.ops.iree


@cat.override(AllOfType(Tensor, PrimitiveTensor))
def cat_default(tensors: Sequence[Tensor | PrimitiveTensor], dim: int):
    return torch.cat([unbox_tensor(t) for t in tensors], dim)


# conv2d


def conv2d_default(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    *,
    stride,
    padding,
    dilation,
    groups,
    accum_dtype: Optional[torch.dtype],
):
    input = unbox_tensor(input)
    weight = unbox_tensor(weight)
    if bias is not None:
        bias = unbox_tensor(bias)
    if weight.dtype != input.dtype:
        weight = weight.to(input.dtype)
    if bias is not None and bias.dtype != input.dtype:
        bias = bias.to(input.dtype)
    return F.conv2d(
        input,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


conv2d.override(Tensor, Tensor, Tensor, auto_dequant=True)(conv2d_default)
conv2d.override(Tensor, Tensor, auto_dequant=True)(conv2d_default)

# Einsum
def mk_menk_men(inputs, weights):
    # batch dims: m, lhs pdims: none, lhs rdims: k, rhs pdims: en, rhs rdims: k
    inputs = inputs.unsqueeze(1)
    weights_shape = weights.shape
    weights = weights.view(
        weights_shape[0], weights_shape[1] * weights_shape[2], weights_shape[3]
    )
    result = matmul(inputs, weights, transpose_rhs=True)
    result = result.view(weights_shape[0], weights_shape[1], weights_shape[2])
    return result


def mek_menk_men(inputs, weights):
    # batch dims: me, lhs pdims: none, lhs rdims: k, rhs pdims: n, rhs rdims: k
    inputs_shape = inputs.shape
    inputs = inputs.view(inputs_shape[0] * inputs_shape[1], 1, inputs_shape[2])
    weights_shape = weights.shape
    weights = weights.view(
        weights_shape[0] * weights_shape[1], weights_shape[2], weights_shape[3]
    )
    result = matmul(inputs, weights, transpose_rhs=True)
    result = result.view(weights_shape[0], weights_shape[1], weights_shape[2])
    return result


def me_men_men(inputs, weights):
    # batch dims: me, lhs pdims: none, lhs rdims: none, rhs pdims: n, rhs rdims: none
    inputs_shape = inputs.shape
    inputs = inputs.view(inputs_shape[0] * inputs_shape[1], 1, 1)
    weights_shape = weights.shape
    weights = weights.view(weights_shape[0] * weights_shape[1], weights_shape[2], 1)
    result = matmul(inputs, weights, transpose_rhs=True)
    result = result.view(weights_shape[0], weights_shape[1], weights_shape[2])
    return result


@einsum_2args.override(AllOfType(Tensor, PrimitiveTensor, QuantizedTensor))
def einsum_2args(input0, input1, einsum_str):
    # Special optimized einsum kernels that lower to batch matmul
    if einsum_str == "mk,menk->men":
        return mk_menk_men(input0, input1)
    elif einsum_str == "mek,menk->men":
        return mek_menk_men(input0, input1)
    elif einsum_str == "me,men->men":
        return me_men_men(input0, input1)
    # Default non-QuantizedTensor einsum
    if not isinstance(input1, QuantizedTensor):
        return torch.einsum(einsum_str, unbox_tensor(x), unbox_tensor(y))
    # Fallback to other kernels
    return NotImplemented


# Elementwise
@elementwise.override(Tensor)
def elementwise_unary(operator, x, *args, **kwargs):
    x = unbox_tensor(x)
    return operator(x, *args, **kwargs)


@elementwise.override(
    AllOfExprs(
        IsOfType(Tensor, PrimitiveTensor), IsOfType(Tensor, PrimitiveTensor, Number)
    )
)
def elementwise_binary(operator, x, y, *args, **kwargs):
    x = unbox_tensor(x)
    if isinstance(y, PrimitiveTensor):
        y = unbox_tensor(y)
    return operator(x, y, *args, **kwargs)


@elementwise.override(
    AllOfExprsVariadic(
        IsOfType(Tensor, InferenceTensor),
        IsOfType(Tensor, InferenceTensor, Number),
        IsOfType(Tensor, InferenceTensor, Number),
    )
)
def elementwise_variadic(operator, x, y, *args):
    """Folds by successively applying the binary operator from left to right until
    exhaustion.

    Match a variable number of tensor/number arguments with at least 3 such arguments.

    Example matches
    ```
    (Tensor, Tensor, Tensor)
    (Tensor, DefaultPrimitiveTensor, float),
    (SplitPrimitiveTensor, ReplicatedTensor, int, Tensor)
    ```

    Will not match
    ```
    (Tensor)
    (Tensor, Tensor)
    (int, Tensor, Tensor)
    ```
    """
    res = elementwise(operator, x, y)
    for arg in args:
        res = elementwise(operator, res, arg)
    return res


# Embedding Lookup
@embedding_lookup.override(Tensor, Tensor)
def embedding_lookup_default(input, embedding_matrix, dtype: dtype):
    return F.embedding(unbox_tensor(input), unbox_tensor(embedding_matrix).to(dtype))


@embedding_lookup.override(Tensor, QuantizedTensor)
def embedding_lookup_Tensor_QuantizedTensor(
    input, embedding_matrix: QuantizedTensor, dtype: dtype
):
    dequant = embedding_matrix.unpack().dequant(dtype=dtype)
    return F.embedding(unbox_tensor(input), dequant)


@equal.override(Tensor, Tensor)
def equal_default(a, b) -> bool:
    return torch.equal(unbox_tensor(a), unbox_tensor(b))


@expand.override(Tensor)
def expand_default(tensor: AnyTensor, shape: List[int]) -> AnyTensor:
    return unbox_tensor(tensor).expand(*shape)


@flatten.override(Tensor)
def flatten_default(
    input: Union[Tensor, PrimitiveTensor], start_dim: int, end_dim: int
) -> Tensor:
    return torch.flatten(unbox_tensor(input), start_dim, end_dim)


@gather.override(Tensor, Tensor)
def gather_default(
    input: Union[Tensor, PrimitiveTensor],
    dim: int,
    index: Union[Tensor, PrimitiveTensor],
) -> Tensor:
    return torch.gather(unbox_tensor(input), dim, unbox_tensor(index))


@get_index.override(AllOfType(Tensor, PrimitiveTensor))
def get_index_default(tensor, key):
    return unbox_tensor(tensor).__get_item__(key)


@get_index.override(QuantizedTensor)
def get_index_QuantizedTensor(tensor: QuantizedTensor, key: slice):
    unpacked = tensor.unpack()
    if isinstance(unpacked, BlockScaledI4Layout):
        mul = 2
    else:
        return NotImplemented
    new_d = unpacked._d[key]
    new_qs = unpacked._qs[key]
    if unpacked.m is not None:
        new_m = unpacked.m[key]
    dims = new_qs.shape
    dims = dims[:-2] + (dims[-2] * dims[-1] * mul,)
    layout = BlockScaledI4Layout(shape=dims, d=new_d, qs=new_qs, m=new_m)
    return PlanarQuantizedTensor(shape=dims, layout=layout)


@gemm.override(AllOfType(Tensor, InferenceTensor))
def gemm(
    a: AnyTensor,
    b: AnyTensor,
    c: Optional[AnyTensor],
    alpha: Optional[Union[Number, AnyTensor]],
    beta: Optional[Union[Number, AnyTensor]],
    transa: bool,
    transb: bool,
) -> bool:
    if transa:
        a = a.T
    if transb:
        b = b.T
    res = matmul(a, b)
    if alpha is not None:
        res = alpha * res
    if c is not None:
        if beta is not None:
            res = res + beta * c
        else:
            res = res + c
    return res


# Group norm.
@group_norm_affine.override(Tensor, Tensor, Tensor)
def group_norm_affine_default(input, weight, bias, *, num_groups, eps):
    input = unbox_tensor(input)
    weight = unbox_tensor(weight)
    bias = unbox_tensor(bias)
    return F.group_norm(input, num_groups=num_groups, weight=weight, bias=bias, eps=eps)


@index_copy_.override(Tensor, Tensor, Tensor)
def index_copy__default(
    inout: Union[Tensor, PrimitiveTensor],
    dim: int,
    index: Union[Tensor, PrimitiveTensor],
    tensor: Union[Tensor, PrimitiveTensor],
) -> Union[Tensor, PrimitiveTensor]:
    unbox_tensor(inout).index_copy_(dim, unbox_tensor(index), unbox_tensor(tensor))
    return inout


@index_put_.override(AllOfType(Tensor, PrimitiveTensor))
def index_put__default(
    inout: Union[Tensor, PrimitiveTensor],
    indices: Tuple[Union[Tensor, PrimitiveTensor]],
    values: Union[Tensor, PrimitiveTensor],
) -> Union[Tensor, PrimitiveTensor]:
    indices = tuple(unbox_tensor(index) for index in indices)
    unbox_tensor(inout).index_put_(indices, unbox_tensor(values))
    return inout


@index_select.override(Tensor, Tensor)
def index_select_default(
    tensor: Union[Tensor, PrimitiveTensor],
    dim: int,
    index: Union[Tensor, PrimitiveTensor],
) -> Union[Tensor, PrimitiveTensor]:
    return torch.index_select(unbox_tensor(tensor), dim, unbox_tensor(index))


@interpolate.override(Tensor)
def interpolate_default(
    input: Tensor,
    size: Optional[int | List[int]],
    scale_factor: Optional[float | List[float]],
    mode: str,
    align_corners: Optional[bool],
    recompute_scale_factor: Optional[bool],
    antialias: bool,
) -> Tensor:
    return torch.nn.functional.interpolate(
        input=unbox_tensor(input),
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
        antialias=antialias,
    )


@layer_norm.override(Tensor, Tensor, Tensor)
def layer_norm_default(input, weight, bias, *, eps):
    input = unbox_tensor(input)
    weight = unbox_tensor(weight)
    bias = unbox_tensor(bias)
    return F.layer_norm(
        input, normalized_shape=weight.shape, weight=weight, bias=bias, eps=eps
    )


# Linear
def linear_default(input, weight, bias, *, accum_dtype) -> Tensor:
    input = unbox_tensor(input)
    weight = unbox_tensor(weight)
    bias = None if bias is None else unbox_tensor(bias)
    if weight.dtype != input.dtype:
        weight = weight.to(dtype=input.dtype)
    result = matmul(input, weight, transpose_rhs=True)
    if bias is not None:
        result = result + bias
    return result


linear.override(Tensor, Tensor, auto_dequant=True)(linear_default)
linear.override(Tensor, Tensor, Tensor, auto_dequant=True)(linear_default)


# Matmul
@matmul.override(Tensor, Tensor, auto_dequant=True)
def matmul_default(lhs, rhs, *, transpose_rhs: bool) -> Tensor:
    lhs = unbox_tensor(lhs)
    rhs = unbox_tensor(rhs)
    if transpose_rhs:
        rhs = rhs.mT
    return torch.matmul(lhs, rhs.to(lhs.dtype))


# Scaled dot product attention
@scaled_dot_product_attention.override(
    Tensor, Tensor, Tensor, Optional[Tensor], auto_dequant=True
)
def scaled_dot_product_attention(q, k, v, a) -> Tensor:
    q = unbox_tensor(q)
    k = unbox_tensor(k)
    v = unbox_tensor(v)
    if a is not None:
        a = unbox_tensor(a)

    # TODO: plumb dropout and is_causal through ops
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=a, dropout_p=0.0, is_causal=False
    )


@mean.override(Tensor)
def mean_default(
    x: Tensor, dim: Union[int, List[int]], keepdim: bool, *, dtype: torch.dtype
) -> None:
    return torch.mean(unbox_tensor(x), dim=dim, keepdim=keepdim, dtype=dtype)


@module_register_buffer.override(torch.nn.Module, Tensor)
def module_register_buffer_default(
    module: torch.nn.Module, name: str, tensor: Union[Tensor, InferenceTensor]
) -> None:
    return module.register_buffer(name, unbox_tensor(tensor))


@repeat.override(Tensor)
def repeat_default(input: Union[Tensor, PrimitiveTensor], *sizes: List[int]) -> Tensor:
    return unbox_tensor(input).repeat(*sizes)


@reshape.override(Tensor)
def reshape_default(input: Union[PrimitiveTensor, Tensor], shape: List[int]) -> Tensor:
    return torch.reshape(unbox_tensor(input), shape)


# RMS norm
@rms_norm.override(AllOfType(Tensor, InferenceTensor))
def rms_norm_default(x, weight, *, epsilon: float) -> Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    output = x * elementwise(torch.rsqrt, variance + epsilon)
    # The cast here is to match the hf implementation, affects numerics
    output = weight * to(output, weight.dtype)
    return output


@rms_norm.override(Tensor, QuantizedTensor)
def rms_norm_Tensor_QuantizedTensor(
    x, weight: PrimitiveTensor, *, epsilon: float
) -> Tensor:
    x = unbox_tensor(x)
    weight = weight.unpack().dequant(x.dtype)
    return rms_norm_default(x, weight, epsilon=epsilon)


@permute.override(Tensor)
def permute(tensor: Tensor, dims: List[int]):
    torch_tensor = unbox_tensor(tensor)
    return torch.permute(torch_tensor, dims)


@softmax.override(Tensor)
def softmax_default(
    tensor: Union[Tensor, PrimitiveTensor],
    dim: Optional[int],
    dtype: Optional[torch.dtype],
) -> Tensor:
    return F.softmax(unbox_tensor(tensor), dim=dim, dtype=dtype)


@to.override(Tensor)
def to_default(tensor: Tensor, *args, **kwargs):
    return unbox_tensor(tensor).to(*args, **kwargs)


@transfer_to_logical_device.override(Tensor)
def transfer_to_logical_device_default(tensor: Tensor, ordinal: int):
    return iree.turbine.ops.iree.transfer_to_logical_device(
        f"{ordinal}", unbox_tensor(tensor)
    )


@transpose.override(Tensor)
def transpose_default(
    tensor: Union[Tensor, PrimitiveTensor], dim0: int, dim1: int
) -> Tensor:
    return torch.transpose(unbox_tensor(tensor), dim0, dim1)


# Sharded default impls (do nothing).


@sharded_cat.override(Tensor)
def sharded_cat_unsharded(maybe_sharded):
    return unbox_tensor(maybe_sharded)


@sharded_sum.override(Tensor)
def sharded_sum_unsharded(maybe_sharded):
    return unbox_tensor(maybe_sharded)


@unflatten.override(Tensor)
def unflatten_default(
    input: Union[Tensor, PrimitiveTensor], dim: int, sizes: Tuple[int]
) -> Tensor:
    return torch.unflatten(unbox_tensor(input), dim, sizes)


@unsqueeze.override(Tensor)
def unsqueeze_default(tensor: Union[Tensor, PrimitiveTensor], dim: int) -> Tensor:
    return torch.unsqueeze(tensor, dim)


@view.override(Tensor)
def view_default(tensor: Union[Tensor, PrimitiveTensor], shape: List[int]) -> Tensor:
    return unbox_tensor(tensor).view(*shape)


@view.override(QuantizedTensor)
def view_QuantizedTensor(tensor: QuantizedTensor, shape):
    unpacked = tensor.unpack()
    if not isinstance(unpacked, BlockScaledI4Layout):
        return NotImplemented
    bs = 16
    shape = list(shape)
    new_d = unpacked._d.view(shape[:-1] + [shape[-1] // 32, 1])
    qs_shape = shape[:-1] + [shape[-1] // 32, 16]
    new_qs = unpacked._qs.view(qs_shape)
    if unpacked.m is not None:
        new_m = unpacked.m.view(shape[:-1] + [shape[-1] // 32, 1])
    layout = BlockScaledI4Layout(shape=shape, d=new_d, qs=new_qs, m=new_m)
    return PlanarQuantizedTensor(shape=shape, layout=layout)
