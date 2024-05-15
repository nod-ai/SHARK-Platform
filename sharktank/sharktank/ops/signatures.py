# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Signatures for dynamic dispatch of ops covering our fundamental tensor types."""

import torch
from torch import Tensor, dtype

from ._registry import *

__all__ = [
    "embedding_lookup",
    "matmul",
    "rms_norm",
]


@overridable
def embedding_lookup(
    input: AnyTensor, embedding_matrix: AnyTensor, dtype: dtype
) -> AnyTensor:
    """Performs the equivalent of F.embedding(input, embedding_matrix).

    Note that the default algorithm will unquantize the embedding_matrix to
    do the lookup, which is inefficient. Specializations should decompose
    this as appropriate for quantized arithmetic.
    """
    raise NotImplementedError


@embedding_lookup.trampoline
def _embedding_lookup_trampoline(
    d: SignatureDispatcher, input: AnyTensor, embedding_matrix: AnyTensor, dtype: dtype
):
    tensors = (input, embedding_matrix)
    for override in d.find_overrides(tensors):
        result = override(input, embedding_matrix, dtype)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def matmul(lhs: AnyTensor, rhs: AnyTensor, *, transpose_rhs: bool = True):
    """Performs a matmul where the RHS may be an InferenceTensor.

    Unlike torch.matmul, this variant is optimized for emission of a fused
    `matmul(lhs, rhs.T)` and the `transpose_rhs=` defaults to True, indicating
    the the RHS is expected to have been transposed already (by some outside
    force). Most inference optimizers will store their weights in this way
    and assume fusions that operate on them, so we just make it the default.

    Args:
    lhs: Left hand side tensor. Can have dimensionality > 2 for batch.
    rhs: Right hand side tensor. Must be 2d.
    transpose_rhs: Whether the right hand side should be transposed prior
        to matmul.
    """
    raise NotImplementedError


@matmul.trampoline
def _matmul_trampoline(d: SignatureDispatcher, lhs, rhs, *, transpose_rhs: bool = True):
    tensors = (lhs, rhs)
    for override in d.find_overrides(tensors):
        result = override(lhs, rhs, transpose_rhs=transpose_rhs)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def rms_norm(x: AnyTensor, weight: AnyTensor, *, epsilon: float) -> AnyTensor:
    """Computes the full, unbiased RMS normalization of an input."""
    raise NotImplementedError


@rms_norm.trampoline
def _rms_norm_trampoline(
    d: SignatureDispatcher, x: AnyTensor, weight: AnyTensor, *, epsilon: float
):
    tensors = (x, weight)
    for override in d.find_overrides(tensors):
        result = override(x, weight, epsilon=epsilon)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)
