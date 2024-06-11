# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Implementations for the q* family of ops.

Note that unlike the normal FP ops, these ops are only parameterized on
distinct types that have a direct implementation in the system. They are
non general (and do not fall back to generic unboxing/dequant). It is
expected that layers which use them know specifically that they wish to
perform some flavor of fully quantized arithmetic.

This module does not contain op overrides that are merely *optimizations*
of a more general algorithm. Practically, this means that auto-dequantizing
ops (which includes weight-only quantizations) happen in custom_impls and
fully quantized op implementations are defined here.
"""

from typing import Optional

import torch

from ..types import (
    InferenceTensor,
    PrimitiveTensor,
    QuantizedTensor,
    TensorScaledLayout,
)

from ._registry import unbox_tensor, AnyTensor
from .signatures import *

################################################################################
# qlinear
################################################################################


def qlinear_dequant_tensor_scaled(
    x: QuantizedTensor,
    weight: QuantizedTensor,
    bias: Optional[AnyTensor],
    *,
    accum_dtype: torch.dtype
) -> torch.Tensor:
    if not issubclass(x.layout_type, TensorScaledLayout) or not issubclass(
        weight.layout_type, TensorScaledLayout
    ):
        return NotImplemented

    # Now we know that both the x/weight are TensorScaledLayout. There are still
    # degrees of freedom:
    #  * Either/both can be per-tensor or per-axis scaled (d is 0D or d is nd>0).
    #  * Either/both can have offsets of not (m is not None).
    x_layout: TensorScaledLayout = x.unpack()
    weight_layout: TensorScaledLayout = weight.unpack()

    # Alias components (d=scale, qs=quantized samples, m=offset)
    x_d = x_layout.d
    x_qs = x_layout.qs
    x_m = x_layout.m
    weight_d = weight_layout.d
    weight_qs = weight_layout.qs
    weight_m = weight_layout.m

    # TODO: Handle permutation that we have a kernel for.

    # Fall back to exact simulation using higher precision types.
    # Note that if implemented in a kernel, the offsets ('m') would be applied
    # to each dot-product row. However, since working layerwise, we just
    # apply them after promoting to the higher precision type. These are
    # mathematically equivalent but not necessarily performance equivalent.
    x_qs = x_qs.to(accum_dtype)
    if x_m is not None:
        x_qs = x_qs - x_m
    weight_qs = weight_qs.to(accum_dtype)
    if weight_m is not None:
        weight_qs = weight_qs - weight_m
    y_qs = torch.matmul(x_qs, weight_qs.T)
    # Output scale by the product of input and weight scale.
    rescale = x_d * weight_d.T
    y = y_qs.to(rescale.dtype) * rescale

    # In this variant, the bias is always full precision, not quantized.
    bias = None if bias is None else unbox_tensor(bias)
    if bias is not None:
        y = y + bias
    return y


# Overrload for both bias and no bias.
qlinear_dequant.override(QuantizedTensor, QuantizedTensor)(
    qlinear_dequant_tensor_scaled
)
qlinear_dequant.override(QuantizedTensor, QuantizedTensor, AnyTensor)(
    qlinear_dequant_tensor_scaled
)
