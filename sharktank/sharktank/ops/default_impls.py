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
