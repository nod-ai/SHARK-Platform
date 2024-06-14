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
from .signatures import (
    IntOrSequenceInt,
    conv2d,
    elementwise,
)


def qconv2d_tensor_scaled_integer(
    input: QuantizedTensor,
    weight: QuantizedTensor,
    bias: Optional[AnyTensor] = None,
    *,
    stride: IntOrSequenceInt = 1,
    padding: IntOrSequenceInt = 0,
    dilation: IntOrSequenceInt = 1,
    groups: IntOrSequenceInt = 1,
    accum_dtype: torch.dtype,
):
    # Only handle tensor scaled layouts.
    if not issubclass(input.layout_type, TensorScaledLayout) or not issubclass(
        weight.layout_type, TensorScaledLayout
    ):
        return NotImplemented

    # Now we know that both the x/weight are TensorScaledLayout. There are still
    # degrees of freedom:
    #  * Either/both can be per-tensor or per-axis scaled (d is 0D or d is nd>0).
    #  * Either/both can have offsets of not (m is not None).
    input_layout: TensorScaledLayout = input.unpack()
    weight_layout: TensorScaledLayout = weight.unpack()

    # Only handle integer quantizations.
    if (
        input_layout.qs.dtype.is_floating_point
        or weight_layout.qs.dtype.is_floating_point
    ):
        return NotImplemented

    # Bias is both optional and may either be quantized or fp.
    bias_qs = None
    rescale_d = None
    if bias is not None:
        if isinstance(bias, QuantizedTensor):
            bias_layout: TensorScaledLayout = bias.unpack()
            if isinstance(bias_layout, TensorScaledLayout):
                bias_qs = bias_layout.qs
                # If a quantized bias is provided, use its scale as the output scale and
                # add directly in integer. A quantized bias cannot be arbitrary and must
                # be a symmetric quantization of the output scale. This is not verified
                # and driven by the data.
                # Broadcast the bias scale to the channels in the NCHW output.
                rescale_d = bias_layout.d.reshape(1, -1, 1, 1)
            else:
                warnings.warn(f"unsupported qconv bias quantization: {bias_layout}")

    # No quantized bias: infer the output rescale.
    if rescale_d is None:
        # Output scale by the product of input and weight scale and broadcast
        # to the output NCHW shape.
        rescale_d = (input_d * flat_weight_d).reshape(1, -1, 1, 1)

    # Alias components (d=scale, qs=quantized samples, m=offset).
    if accum_dtype is None:
        accum_dtype = torch.int32
    input_d = input_layout.d
    input_dtype = input_layout.dtype
    input_qs = input_layout.qs
    input_m = input_layout.m
    weight_d = weight_layout.d
    weight_qs = weight_layout.qs
    weight_m = weight_layout.m

    # Verify that quantization axis meets our requirements.
    input_d_shape = input_d.shape
    input_d_rank = len(input_d_shape)
    weight_d_shape = weight_d.shape
    weight_d_rank = len(weight_d_shape)
    if input_d_rank > 0:
        # Presently only support per-tensor quantization of input. Others are
        # possible but need to be use-case driven/verified.
        return NotImplemented
    if weight_d_rank == 0:
        # Per-tensor weight supported.
        flat_weight_d = weight_d
    elif (
        weight_d_rank == 4
        and weight_d_shape[1] == 1
        and weight_d_shape[2] == 1
        and weight_d_shape[3] == 1
    ):
        # Per output channel supported.
        flat_weight_d = weight_d.flatten()
    else:
        # Others not supported.
        return NotImplemented

    # TODO: Use a real mixed precision op.
    y_qs = torch.nn.functional.conv2d(
        input_qs.to(dtype=torch.float32),
        weight_qs.to(dtype=torch.float32),
        bias=bias_qs.to(dtype=torch.float32),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    ).to(dtype=accum_dtype)

    output_shape = list(y_qs.shape)
    y = PlanarQuantizedTensor(
        shape=output_shape,
        layout=TensorScaledLayout(
            shape=output_shape,
            d=rescale_d,
            qs=y_qs,
            dtype=input_dtype,
        ),
    )

    # If we have an unquantized bias, dequantize the result and add here.
    if bias is not None and bias_qs is None:
        y = y.unpack().dequant()
        y = elementwise(torch.add, y, bias)
    return y


conv2d.override(QuantizedTensor, QuantizedTensor)(qconv2d_tensor_scaled_integer)
conv2d.override(QuantizedTensor, QuantizedTensor, AnyTensor)(
    qconv2d_tensor_scaled_integer
)
