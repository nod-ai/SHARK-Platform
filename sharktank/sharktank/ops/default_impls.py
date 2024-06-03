# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file contains overrides of the standard ops for normal torch and
# generic primitive/quantized types.

import torch
from torch import Tensor, dtype
import torch.nn.functional as F

from ..types import InferenceTensor, PrimitiveTensor, QuantizedTensor
from ._registry import unbox_tensor
from .signatures import *

# conv2d
@conv2d.override(Tensor, Tensor, Tensor)
def conv2d_with_bias(
    input: Tensor, weight: Tensor, bias: Tensor, *, stride, padding, dilation, groups
):
    input = unbox_tensor(input)
    weight = unbox_tensor(weight)
    bias = unbox_tensor(bias)
    if weight.dtype != input.dtype:
        weight = weight.to(input.dtype)
    if bias.dtype != input.dtype:
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


@conv2d.override(Tensor, Tensor)
def conv2d_no_bias(
    input: Tensor, weight: Tensor, bias, *, stride, padding, dilation, groups
):
    assert bias is None
    input = unbox_tensor(input)
    weight = unbox_tensor(weight)
    if weight.dtype != input.dtype:
        weight = weight.to(input.dtype)
    return F.conv2d(
        input,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


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


# Matmul
@matmul.override(Tensor, Tensor)
def matmul_default(lhs, rhs, *, transpose_rhs: bool) -> Tensor:
    lhs = unbox_tensor(lhs)
    rhs = unbox_tensor(rhs)
    if transpose_rhs:
        rhs = rhs.T
    return torch.matmul(lhs, rhs.to(lhs.dtype))


@matmul.override(Tensor, QuantizedTensor)
def matmul_Tensor_QuantizedTensor(
    lhs, rhs: QuantizedTensor, *, transpose_rhs: bool
) -> Tensor:
    lhs = unbox_tensor(lhs)
    rhs_torch = rhs.unpack().dequant(lhs.dtype)
    return matmul_default(lhs, rhs_torch, transpose_rhs=transpose_rhs)


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


# Sharded default impls (do nothing).


@sharded_cat.override(Tensor)
def sharded_cat_unsharded(maybe_sharded):
    return unbox_tensor(maybe_sharded)


@sharded_sum.override(Tensor)
def sharded_sum_unsharded(maybe_sharded):
    return unbox_tensor(maybe_sharded)
