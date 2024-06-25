# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file contains overrides of the standard ops for normal torch and
# generic primitive/quantized types.

from typing import Optional, List

import torch
from torch import Tensor, dtype
import torch.nn.functional as F

from ..types import InferenceTensor, PrimitiveTensor, QuantizedTensor
from ._registry import unbox_tensor, AnyTensor
from .signatures import *

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

# Elementwise
@elementwise.override(Tensor)
def elementwise_unary(operator, x):
    x = unbox_tensor(x)
    return operator(x)


@elementwise.override(Tensor, Tensor)
def elementwise_binary(operator, x, y):
    x = unbox_tensor(x)
    y = unbox_tensor(y)
    return operator(x, y)


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


# Group norm.
@group_norm_affine.override(Tensor, Tensor, Tensor)
def group_norm_affine_default(input, weight, bias, *, num_groups, eps):
    input = unbox_tensor(input)
    weight = unbox_tensor(weight)
    bias = unbox_tensor(bias)
    return F.group_norm(input, num_groups=num_groups, weight=weight, bias=bias, eps=eps)


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
        rhs = rhs.T
    return torch.matmul(lhs, rhs.to(lhs.dtype))


# RMS norm
@rms_norm.override(Tensor, Tensor)
def rms_norm_default(x, weight, *, epsilon: float) -> Tensor:
    x = unbox_tensor(x)
    weight = unbox_tensor(weight)
    variance = x.pow(2).mean(-1, keepdim=True)
    output = x * torch.rsqrt(variance + epsilon)
    output = output * weight
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


# Sharded default impls (do nothing).


@sharded_cat.override(Tensor)
def sharded_cat_unsharded(maybe_sharded):
    return unbox_tensor(maybe_sharded)


@sharded_sum.override(Tensor)
def sharded_sum_unsharded(maybe_sharded):
    return unbox_tensor(maybe_sharded)
