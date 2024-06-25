# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Signatures for dynamic dispatch of ops covering our fundamental tensor types."""

from typing import Optional, Sequence, Union, List

import torch
import numbers
from torch import Tensor, dtype
from ..types import ShardedTensor

from ._registry import *

__all__ = [
    "all_gather",
    "conv2d",
    "elementwise",
    "embedding_lookup",
    "equal",
    "group_norm_affine",
    "layer_norm",
    "linear",
    "matmul",
    "permute",
    "rms_norm",
    "replicate",
    "reshard_split",
    "reshard_like",
    "sharded_cat",
    "sharded_sum",
]

IntOrSequenceInt = Union[int, Sequence[int]]


@overridable
def all_gather(maybe_sharded: AnyTensor, *, dim: int | None = None) -> AnyTensor:
    ...


@all_gather.trampoline
def _all_gather_trampoline(
    d: SignatureDispatcher, maybe_sharded: AnyTensor, *, dim: int | None = None
):
    tensors = (maybe_sharded,)
    for override in d.find_overrides(tensors):
        result = override(maybe_sharded, dim=dim)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def conv2d(
    input: AnyTensor,
    weight: AnyTensor,
    bias: Optional[AnyTensor] = None,
    *,
    stride: IntOrSequenceInt = 1,
    padding: IntOrSequenceInt = 0,
    dilation: IntOrSequenceInt = 1,
    groups: IntOrSequenceInt = 1,
    accum_dtype: Optional[torch.dtype] = None,
):
    """Equivalent to torch.nn.functional.conv2d with enhancements:

    * Primitive weight/bias tensors will be promoted to the input dtype.
    """
    raise NotImplementedError


@conv2d.trampoline
def _conv2d_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    weight: AnyTensor,
    bias: Optional[AnyTensor] = None,
    *,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    accum_dtype: Optional[torch.dtype] = None,
):
    tensors = [input, weight]
    if bias is not None:
        tensors.append(bias)
    for override in d.find_overrides(tensors):
        result = override(
            input,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            accum_dtype=accum_dtype,
        )
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def elementwise(operator, *args: AnyTensor) -> AnyTensor:
    """Applies an elementwise operator against arguments."""
    raise NotImplementedError


@elementwise.trampoline
def _elementwise_trampoline(d: SignatureDispatcher, operator, *args: AnyTensor):
    tensors = args
    for override in d.find_overrides(tensors):
        result = override(operator, *args)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


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
def equal(a: AnyTensor, b: AnyTensor) -> bool:
    """Compares 2 tensors for equality, such that if one is substituted with the other
    in sharktank polymorphic calls, the results will be essentially the same.
    Meaning, they would also compare equal.

    Overrides are matched first against both tensor types and failing that,
    then on just the first.
    Therefore, each first-only argument override must internally decide whether
    it can handle an equality check with an arbitrary b tensor.

    torch.Tensor and DefaultPrimitiveTensor with the same contents would compare equal."""
    ...


@equal.trampoline
def _equal_trampoline(d: SignatureDispatcher, a: AnyTensor, b: AnyTensor):
    # Try first more specific matching the 2 operands.
    tensors = (
        a,
        b,
    )
    for override in d.find_overrides(tensors):
        result = override(a, b)
        if result is not NotImplemented:
            return override, result

    # Less specific. Try matching only the first operand.
    tensors = (a,)
    for override in d.find_overrides(tensors):
        result = override(a, b)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def group_norm_affine(
    input: AnyTensor, weight: AnyTensor, bias: AnyTensor, *, num_groups: int, eps: float
):
    """Equivalent to torch.nn.functional.group_norm(affine=True)."""
    raise NotImplementedError


@group_norm_affine.trampoline
def _group_norm_affine_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    weight: AnyTensor,
    bias: AnyTensor,
    *,
    num_groups: int,
    eps: float,
):
    tensors = (input, weight, bias)
    for override in d.find_overrides(tensors):
        result = override(input, weight, bias, num_groups=num_groups, eps=eps)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def layer_norm(
    input: AnyTensor, weight: AnyTensor, bias: Optional[AnyTensor], *, eps: float
):
    """Equivalent to torch.nn.functional.layer_norm(elementwise_affine=True)."""
    raise NotImplementedError


@layer_norm.trampoline
def _layer_norm_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    weight: AnyTensor,
    bias: Optional[AnyTensor],
    *,
    eps: float,
):
    tensors = [input, weight]
    if bias is not None:
        tensors.append(bias)
    for override in d.find_overrides(tensors):
        result = override(input, weight, bias, eps=eps)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def linear(
    input: AnyTensor,
    weight: AnyTensor,
    bias: Optional[AnyTensor] = None,
    *,
    accum_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Applies a linear transformation to the incoming data.

    Equivalent to:
    ```
    y = torch.matmul(input, weight.T) + bias
    ```

    This operator is defined to operate on a limited number of quantized types.
    In that situation, the result may be a QuantizedTensor. Callers should
    be prepared to handle this scenario.

    The optional accum_dtype argument is used as a hint to some implementations
    which may need help in selecting an appropriate high precision type for
    accumulation.
    """
    raise NotImplementedError


@linear.trampoline
def _linear_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    weight: AnyTensor,
    bias: Optional[AnyTensor] = None,
    *,
    accum_dtype: Optional[torch.dtype] = None,
):
    tensors = (input, weight) if bias is None else (input, weight, bias)
    for override in d.find_overrides(tensors):
        result = override(input, weight, bias, accum_dtype=accum_dtype)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def matmul(lhs: AnyTensor, rhs: AnyTensor, *, transpose_rhs: bool = False):
    """Performs a matmul where the RHS may be an InferenceTensor.

    Unlike torch.matmul, this variant is optimized for emission of a fused
    `matmul(lhs, rhs.T)` and the `transpose_rhs=` defaults to True, indicating
    the the RHS is expected to have been transposed already (by some outside
    force). Most inference optimizers will store their weights in this way
    and assume fusions that operate on them, so we just make it the default.

    Args:
    lhs: Left hand side tensor. Can have dimensionality > 2 for batch.
    rhs: Right hand side tensor. Must be 2d or a scalar.
    transpose_rhs: Whether the right hand side should be transposed prior
        to matmul.
    """
    raise NotImplementedError


@matmul.trampoline
def _matmul_trampoline(
    d: SignatureDispatcher, lhs, rhs, *, transpose_rhs: bool = False
):
    tensors = (lhs, rhs)
    assert isinstance(rhs, numbers.Number) or len(rhs.shape) == 2
    for override in d.find_overrides(tensors):
        result = override(lhs, rhs, transpose_rhs=transpose_rhs)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def permute(tensor: AnyTensor, dims: List[int]) -> AnyTensor:
    """Permute the tensor dimensions according to the permutation `dims` in line
    notation.
    The semantics are the same as torch.permute."""
    ...


@permute.trampoline
def _permute_trampoline(d: SignatureDispatcher, tensor: AnyTensor, dims: List[int]):
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor, dims)
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


@overridable
def replicate(input: AnyTensor, count: int) -> ShardedTensor:
    """Replicate across devices.

    Possibly reshards if required."""
    ...


@replicate.trampoline
def _replicate_trampoline(
    d: SignatureDispatcher, input: AnyTensor, count: int
) -> ShardedTensor:
    tensors = (input,)
    for override in d.find_overrides(tensors):
        result = override(input, count=count)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def reshard_split(input: AnyTensor, *, dim: int, count: int) -> ShardedTensor:
    """Split `input` along `dim`.
    This does not mean that a sharded tensor is further sharded.
    It is not composition of sharding operations.
    """
    ...


@reshard_split.trampoline
def _shard_trampoline(
    d: SignatureDispatcher, input: AnyTensor, dim: int, count: int
) -> ShardedTensor:
    tensors = (input,)
    for override in d.find_overrides(tensors):
        result = override(input, dim=dim, count=count)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def reshard_like(input: AnyTensor, like: AnyTensor) -> AnyTensor:
    """Shard `input` the same way as `like`.

    This may require expensive resharding."""
    ...


@reshard_like.trampoline
def _reshard_like_trampoline(
    d: SignatureDispatcher, input: AnyTensor, like: AnyTensor
) -> AnyTensor:
    tensors = (
        input,
        like,
    )
    for override in d.find_overrides(tensors):
        result = override(input, like)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def sharded_cat(maybe_sharded: AnyTensor):
    """Concats all shards along the sharding dimension.

    Does nothing if not sharded.
    """
    raise NotImplementedError


@sharded_cat.trampoline
def _sharded_cat_trampoline(d: SignatureDispatcher, maybe_sharded: AnyTensor):
    tensors = (maybe_sharded,)
    for override in d.find_overrides(tensors):
        result = override(maybe_sharded)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def sharded_sum(maybe_sharded: AnyTensor):
    ...


@sharded_sum.trampoline
def _sharded_sum_trampoline(d: SignatureDispatcher, maybe_sharded: AnyTensor):
    tensors = (maybe_sharded,)
    for override in d.find_overrides(tensors):
        result = override(maybe_sharded)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)
