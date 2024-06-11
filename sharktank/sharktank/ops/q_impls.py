# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Implementations for op variants that are fully quantized.
"""

from typing import Optional

import warnings

import torch

from ..types import (
    QuantizedTensor,
    PlanarQuantizedTensor,
    TensorScaledLayout,
)

from ._registry import unbox_tensor, AnyTensor
from .signatures import *

################################################################################
# qlinear
################################################################################


def qlinear_tensor_scaled_integer(
    x: QuantizedTensor,
    weight: QuantizedTensor,
    bias: Optional[QuantizedTensor],
    *,
    accum_dtype: torch.dtype
) -> torch.Tensor:
    # Only handle tensor scaled layouts.
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

    # Only handle integer quantizations.
    if x_layout.qs.dtype.is_floating_point or weight_layout.qs.dtype.is_floating_point:
        return NotImplemented

    # Bias.
    if bias is not None and not issubclass(bias.layout_type, TensorScaledLayout):
        warnings.warn(
            "qlinear_tensor_scaled_integer falling back to generic because bias is not properly quantized"
        )
        return NotImplemented
    bias_layout: Optional[TensorScaledLayout] = (
        bias.unpack() if bias is not None else None
    )

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

    # We don't have a great way to verify that the bias has been scaled
    # properly, and this is just an invariant that it is compatible with
    # the arithmetic that produces the output scale. If we don't have a bias,
    # we compute the output scale explicitly.
    if bias_layout is not None:
        y_qs = y_qs + bias_layout.qs
        rescale_d = bias_layout.d
    else:
        # Output scale by the product of input and weight scale.
        rescale_d = x_d * weight_d.T

    output_shape = list(y_qs)
    return PlanarQuantizedTensor(
        shape=output_shape,
        layout=TensorScaledLayout(
            shape=output_shape,
            d=rescale_d,
            qs=y_qs,
        ),
    )


# Overrload for both bias and no bias.
linear.override(QuantizedTensor, QuantizedTensor)(qlinear_tensor_scaled_integer)
linear.override(QuantizedTensor, QuantizedTensor, QuantizedTensor)(
    qlinear_tensor_scaled_integer
)
