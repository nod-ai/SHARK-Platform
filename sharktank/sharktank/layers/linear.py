# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch

from .. import ops
from .base import Theta, ThetaLayer
from ..types.layout_utils import saturate_cast
from ..types import (
    DynamicScaledQuantizer,
    QuantizedTensor,
    QuantizerTensor,
    StaticScaledQuantizer,
    TensorScaledLayout,
)

__all__ = [
    "LinearLayer",
]


class LinearLayer(ThetaLayer):
    """Linear layer which computes:

    ```
    matmul(x, weight.T) + bias
    ```

    Whether the weight is transposed as part of the calculation can be
    controlled with `transpose_weight=` (default true).

    This layer will operate in one of several modes based on the presence
    of additional tensors in theta:

    Weight-Only Quant Mode:
    -----------------------
    In this mode, the weight and/or bias is a QuantizedTensor but there is
    no activation quantization information (unless if the input happens to
    have already come in from a previous layer as a QuantizedTensor). In this
    case, the default behavior of the matmul op depends on the exact form of
    weight quantization. If a kernel exists to fuse the computation in some
    way, it will be used. Otherwise, the fallback logic will dequantize the
    weight and perform normal FP math. The compiler may still do some fusion
    on this, depending on many things.

    FakeQuant Mode:
    ---------------
    FakeQuant mode is activated if 'fq_input' and 'fq_output' are present.
    If present, they are expected to be of type `QuantizerTensor`. In this mode:

      * Weight and bias will be dequantized if appropriate (typically, these
        will be stored as some form of QuantizedTensor).
      * The input will be quantized and then dequantized according to the
        'fq_input' quantizer.
      * If there is no 'fq_output' quantizer, we fall back to
        "dynamic quantization"
        mode, where we will dynamically compute an estimated output quantizer
        by multiplying the input scale by the weight scale. This only works
        for TensorScaledLayouts and is meant to simulate the fused dynamic
        quantization mode.
      * The output of the linear computation is quantized and then dequantized
        by the 'fq_output' quantizer (whether explicit or computed).

    While this mode is not particularly interesting for production deployments,
    it duplicates closely the fake_quant logic typically used in simulators and
    can help identify bugs and mismatches that can arise from more aggressive
    techniques.

    Native Quant Mode:
    ------------------
    In native quant mode, the goal is to perform quantized arithmetic. The types
    of the inputs/weight/bias and an output quantizer dictate the precise
    parameters of that arithmetic. The only required tensor in this mode is
    either a 'q_output' or a 'dq_output':

    * 'q_output' is used to specify output quantization parameters, and the
      resulting QuantizedTensor is returned directly from the layer. This
      means that any subsequent layer or operations must be prepared to handle
      inputs with the exact quantization output from here.
    * 'dq_output' is used to specify output quantization parameters and signal
      the layer to dequantize back to a high precision tensor for return.
      Downstream consumers will only see the high precision tensor.

    There can also be an optional 'q_input' which specifies that the input to
    the layer should be quantized according to this quantizer. If ommitted,
    then the input to the layer *must* already be a QuantizedTensor (i.e. from
    a producer).

    Any of these quantizers may be a `DynamicScaledQuantizer`:

    * For input quantizers, quantization parameters will be derived dynamically
      based on the actual activation values.
    * For output quantizers, quantization parameters will be estimated
      dynamically based on the input/weight/bias quantizer/type.

    The bias tensor is expected to be either a `QuantizedTensor` with the
    same scaling as the output or a primitive tensor that will be dynamically
    quantized to the output scale for accumulation.
    """

    def __init__(
        self,
        theta: Theta,
        *,
        weight_name: str = "weight",
        bias_name: str = "bias",
        transpose_weight: bool = True,
    ):
        super().__init__(theta)
        self.weight = self.theta_tensor(weight_name)
        self.bias = None
        if bias_name in self.theta.keys:
            self.bias = self.theta_tensor(bias_name)
        self.transpose_weight = transpose_weight

        # Get mode specific tensors.
        self.mode: Optional[str] = None
        self.fq_input: QuantizerTensor = theta.optional_tensor("fq_input")
        self.fq_output: QuantizerTensor = theta.optional_tensor("fq_output")

        if self.fq_input is not None:
            assert self.mode is None, f"Cannot enable fq mode and {self.mode}"
            self.mode = "fq"
            self._validate_fake_quant_mode()

    def forward(self, x):
        mode = self.mode
        weight = self.weight
        bias = self.bias

        # TODO: Implement native quant mode.

        # Regular or fakequant mode.
        if mode is None or mode == "fq":
            # Input conditioning.
            if self.mode == "fq":
                orig_weight_layout = weight.unpack()
                weight = orig_weight_layout.dequant()
                x = self.fq_input.quantize(x)
                orig_x_layout = x.unpack()
                x = orig_x_layout.dequant()
                if isinstance(bias, QuantizedTensor):
                    bias = bias.unpack().dequant()

            # Primary computation.
            y = ops.matmul(x, weight, transpose_rhs=self.transpose_weight)
            if bias is not None:
                y = ops.elementwise(torch.add, y, bias)

            # Output conditioning.
            if self.mode == "fq":
                fq_output = self.fq_output
                if fq_output is None:
                    # Dynamic output quantization.
                    # TODO: With a bit of work, we could create a Quantizer
                    # and use the normal arithmetic flow vs faking it here.
                    # Dynamically compute output quant and simulate.
                    # TODO: Better estimate of the output dynamic quant.
                    reciprocal_scale = orig_weight_layout.d + orig_x_layout.d
                    scale = 1.0 / reciprocal_scale
                    offset = orig_weight_layout.m
                    # Simulate quant.
                    if offset is not None:
                        qs = (y - offset) * scale
                    else:
                        qs = y * scale
                    output_quant_dtype = orig_x_layout.qs.dtype
                    qs = saturate_cast(qs, output_quant_dtype)
                    # Dequant.
                    output_dequant = reciprocal_scale * qs.to(y.dtype)
                    if offset is not None:
                        output_dequant = output_dequant + offset
                    y = output_dequant
                else:
                    # Static output quantization.
                    y = fq_output.quantize(y).unpack().dequant()
            return y

        raise AssertionError(f"LinearLayer unhandled mode")

    def _validate_fake_quant_mode(self):
        fq_input = self.fq_input
        fq_output = self.fq_output
        weight = self.weight
        if not isinstance(fq_input, QuantizerTensor):
            raise (
                f"LinearLayer requires fq_input to be a QuantizerTensor, "
                f"but got: {fq_input}"
            )
        if fq_output is not None:
            if not isinstance(fq_output, QuantizerTensor):
                raise (
                    f"LinearLayer requires fq_output to be a QuantizerTensor, "
                    f"but got: {fq_output}"
                )
        else:
            # Dynamic quant mode requires a *ScaledQuantizer for the input
            # and a TensorScaledLayout weight tensor.
            if not isinstance(
                fq_input, (StaticScaledQuantizer, DynamicScaledQuantizer)
            ):
                raise AssertionError(
                    f"LinearLayer with a fq_output=None requires a "
                    f"[Static|Dynamic]ScaledQuantizer for fq_input, but got: "
                    f"{repr(fq_input)}"
                )
            if not isinstance(weight, QuantizedTensor):
                raise AssertionError(
                    f"LinearLayer with a fq_output=None requires a weight "
                    f"of type QuantizedTensor, but got: {repr(weight)}"
                )
            if not issubclass(weight.layout_type, TensorScaledLayout):
                raise AssertionError(
                    f"LinearLayer with a fq_output=None requires a QuantizedTensor "
                    f"weight with TensorScaledLayout, but got: {weight.layout_type}"
                )
