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
from .signatures import *


# Embedding Lookup


@embedding_lookup.override(Tensor, Tensor)
def embedding_lookup_default(input: Tensor, embedding_matrix: Tensor, dtype: dtype):
    return F.embedding(input, embedding_matrix.to(dtype))


@embedding_lookup.override(Tensor, PrimitiveTensor)
def embedding_lookup_Tensor_PrimitiveTensor(
    input: Tensor, embedding_matrix: PrimitiveTensor, dtype: dtype
):
    return F.embedding(input, embedding_matrix.as_torch(dtype=dtype))


@embedding_lookup.override(Tensor, QuantizedTensor)
def embedding_lookup_Tensor_QuantizedTensor(
    input: Tensor, embedding_matrix: QuantizedTensor, dtype: dtype
):
    dequant = embedding_matrix.unpack().dequant(dtype=dtype)
    return F.embedding(input, dequant)


# Matmul
@matmul.override(Tensor, Tensor)
def matmul_default(lhs: Tensor, rhs: Tensor, *, transpose_rhs: bool) -> Tensor:
    if transpose_rhs:
        rhs = rhs.T
    return torch.matmul(lhs, rhs.to(lhs.dtype))


@matmul.override(Tensor, PrimitiveTensor)
def matmul_Tensor_PrimitiveTensor(
    lhs: Tensor, rhs: PrimitiveTensor, *, transpose_rhs: bool
) -> Tensor:
    rhs_torch = rhs.as_torch(dtype=lhs.dtype)
    return matmul_default(lhs, rhs_torch, transpose_rhs=transpose_rhs)


@matmul.override(Tensor, QuantizedTensor)
def matmul_Tensor_PrimitiveTensor(
    lhs: Tensor, rhs: QuantizedTensor, *, transpose_rhs: bool
) -> Tensor:
    rhs_torch = rhs.unpack().dequant(lhs.dtype)
    return matmul_default(lhs, rhs_torch, transpose_rhs=transpose_rhs)


# RMS norm


@rms_norm.override(Tensor, Tensor)
def rms_norm_default(x: Tensor, weight: Tensor, *, epsilon: float) -> Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    output = x * torch.rsqrt(variance + epsilon)
    output = output * weight
    return output


@rms_norm.override(Tensor, PrimitiveTensor)
def rms_norm_Tensor_PrimitiveTensor(
    x: Tensor, weight: PrimitiveTensor, *, epsilon: float
) -> Tensor:
    weight = weight.as_torch(dtype=x.dtype)
    return rms_norm_default(x, weight, epsilon=epsilon)


@rms_norm.override(Tensor, QuantizedTensor)
def rms_norm_Tensor_QuantizedTensor(
    x: Tensor, weight: PrimitiveTensor, *, epsilon: float
) -> Tensor:
    weight = weight.unpack().dequant(x.dtype)
    return rms_norm_default(x, weight, epsilon=epsilon)
