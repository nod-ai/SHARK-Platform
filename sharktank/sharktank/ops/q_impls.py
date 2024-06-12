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

_CUDA_HACK = True

################################################################################
# qlinear
################################################################################


def qlinear_tensor_scaled_integer(
    x: QuantizedTensor,
    weight: QuantizedTensor,
    bias: Optional[AnyTensor],
    *,
    accum_dtype: torch.dtype,
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
    quantized_bias_accum = False
    if bias is not None:
        if isinstance(bias, QuantizedTensor):
            bias_layout: TensorScaledLayout = bias.unpack()
            if isinstance(bias_layout, TensorScaledLayout):
                quantized_bias_accum = True
            else:
                warnings.warn(f"unsupported qlinear bias quantization: {bias_layout}")

    # Alias components (d=scale, qs=quantized samples, m=offset)
    x_d = x_layout.d
    x_qs = x_layout.qs
    x_m = x_layout.m
    weight_d = weight_layout.d
    weight_qs = weight_layout.qs
    weight_m = weight_layout.m

    # Only implemented for per-tensor or axis-0 quantization.
    x_d_shape = x_d.shape
    if len(x_d.shape) > 0 and x_d_shape[-1] != 1:
        # Should be scalar or per-axis 0. Example: [16, 1]
        return NotImplemented
    weight_d_shape = weight_d.shape
    if len(weight_d_shape) > 0 and weight_d_shape[-1] != 1:
        # Should be scalar or per-axis 0. Example: [16, 1]
        return NotImplemented

    # TODO: Handle permutation that we have a kernel for.

    # Fall back to automatic fusion based on integer, high precision matmul.
    x_qs = x_qs.to(accum_dtype)
    weight_qs = weight_qs.to(accum_dtype)
    # TODO: CUDA doesn't have an implementation of integer matmul. So promote
    # for now.
    if _CUDA_HACK:
        y_qs = torch.matmul(
            x_qs.to(dtype=torch.float32), weight_qs.to(dtype=torch.float32).T
        ).to(dtype=accum_dtype)
    else:
        y_qs = torch.matmul(x_qs, weight_qs.T)

    # Offset correction. By applying the offset correction in post, it is
    # set up to fuse with its consumer, which is already doing additional
    # activation manipulation. Whereas if applied before, it either blocks
    # matmul fusion or fuses additional arithmetic into the O(n^3) operation.
    if x_m is not None:
        # Apply offset correction for asymmetric x.
        # At the time of writing this was not a common case.
        x_offset_fix = torch.sum(weight_qs, axis=0, keepdim=True) * x_m
        y_qs = y_qs - x_offset_fix
    if weight_m is not None:
        # Apply offset correction for asymmetric weight.
        # At the time of writing this was the common case.
        weight_offset_fix = torch.sum(x_qs, axis=-1, keepdim=True) * weight_m.T
        y_qs = y_qs - weight_offset_fix
    if x_m is not None and weight_m is not None:
        # Apply joint offset correction if both x and weight are asymmetric.
        # At the time of writing this was not a common case.
        xweight_offset_fix = x_m * weight_m.T * x_qs.shape[-1]
        y_qs = y_qs + xweight_offset_fix

    # We don't have a great way to verify that the bias has been scaled
    # properly, and this is just an invariant that it is compatible with
    # the arithmetic that produces the output scale. If we don't have a bias,
    # we compute the output scale explicitly.
    if quantized_bias_accum:
        y_qs = y_qs + bias_layout.qs
        rescale_d = bias_layout.d
    else:
        # Output scale by the product of input and weight scale.
        rescale_d = x_d * weight_d.T

    output_shape = list(y_qs)
    y = PlanarQuantizedTensor(
        shape=output_shape,
        layout=TensorScaledLayout(
            shape=output_shape,
            d=rescale_d,
            qs=y_qs,
        ),
    )

    # If we have a bias that we couldn't add while quantized, add it here.
    if bias is not None and not quantized_bias_accum:
        y = y.unpack().dequant()
        y = elementwise(torch.add, y, bias)

    return y


# Overrload for both bias and no bias.
linear.override(QuantizedTensor, QuantizedTensor)(qlinear_tensor_scaled_integer)
linear.override(QuantizedTensor, QuantizedTensor, AnyTensor)(
    qlinear_tensor_scaled_integer
)
