# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Tuple

import torch

from .. import ops
from ..types import *

from .base import Theta, ThetaLayer


__all__ = [
    "Conv2DLayer",
]


class Conv2DLayer(ThetaLayer):
    """Theta based conv2d layer. This assumes weight/bias naming as per the nn.Conv2D
    module ("weight", "bias").

    This layer is being adapted for quantization. See LinearLayer for full docs
    while this is being worked out.
    """

    def __init__(
        self, theta: Theta, padding: Optional[Tuple[int, int]] = None, stride: int = 1
    ):
        super().__init__(theta)
        assert padding is None or len(padding) == 2
        self.padding = padding
        self.stride = stride
        self.dilation = 1
        self.groups = 1

        self.weight = self.theta.tensor("weight")
        self.bias = self.theta.optional_tensor("bias")

        # Input premultiplier.
        self.premul_input = theta.optional_tensor("premul_input")

        # Get mode specific tensors.
        self.mode: Optional[str] = None
        self.fq_input: Optional[QuantizerTensor] = theta.optional_tensor("fq_input")
        self.fq_output: Optional[QuantizerTensor] = theta.optional_tensor("fq_output")
        self.q_input: Optional[QuantizerTensor] = theta.optional_tensor("q_input")
        self.q_output: Optional[QuantizerTensor] = theta.optional_tensor("q_output")
        self.dq_output: Optional[QuantizerTensor] = theta.optional_tensor("dq_output")

        if self.fq_input is not None:
            assert self.mode is None, f"Cannot enable fq mode and {self.mode}"
            self.mode = "fq"
            self._validate_fake_quant_mode()

        if (
            self.q_input is not None
            or self.q_output is not None
            or self.dq_output is not None
        ):
            assert self.mode is None, f"Cannot enable native quant mode and {self.mode}"
            self.mode = "native_quant"
            self._validate_native_quant_mode()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mode = self.mode
        x = input
        weight = self.weight
        bias = self.bias

        if self.premul_input is not None:
            x = ops.elementwise(torch.mul, x, self.premul_input)

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
            y = ops.conv2d(
                x,
                weight=weight,
                bias=bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            # Output conditioning.
            if self.mode == "fq":
                fq_output = self.fq_output
                if fq_output is not None:
                    # Static output quantization.
                    y = fq_output.quantize(y).unpack().dequant()
            return y

        # Native quant mode.
        # TODO: This needs to be completely replumbed into a couple of different
        # fused ops. For the moment, though, we skate by with simulating it
        # via normal ops.
        if mode == "native_quant":
            q_input = self.q_input
            q_output = self.q_output
            dq_output = self.dq_output
            if q_input is not None:
                x = q_input.quantize(x)

            # TODO: We should be calling an explicit qmatmul that can take
            # the output quantizer. For the moment, we ignore dq_output
            # entirely and only act on a q_output.
            # Primary computation.
            y = ops.conv2d(
                x,
                weight=weight,
                bias=bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            if dq_output is not None and not isinstance(
                dq_output, DynamicScaledQuantizer
            ):
                y = dq_output.quantize(y).unpack().dequant()

            if q_output is not None:
                y = q_output.quantize(y)

            return y

        raise AssertionError(f"Conv2DLayer unhandled mode '{mode}'")

    def _validate_fake_quant_mode(self):
        fq_input = self.fq_input
        fq_output = self.fq_output
        weight = self.weight
        if not isinstance(fq_input, QuantizerTensor):
            raise (
                f"Conv2DLayer requires fq_input to be a QuantizerTensor, "
                f"but got: {fq_input}"
            )
        if fq_output is not None:
            if not isinstance(fq_output, QuantizerTensor):
                raise (
                    f"Conv2DLayer requires fq_output to be a QuantizerTensor, "
                    f"but got: {fq_output}"
                )
        else:
            # Dynamic quant mode requires a *ScaledQuantizer for the input
            # and a TensorScaledLayout weight tensor.
            if not isinstance(
                fq_input, (StaticScaledQuantizer, DynamicScaledQuantizer)
            ):
                raise AssertionError(
                    f"Conv2DLayer with a fq_output=None requires a "
                    f"[Static|Dynamic]ScaledQuantizer for fq_input, but got: "
                    f"{repr(fq_input)}"
                )
            if not isinstance(weight, QuantizedTensor):
                raise AssertionError(
                    f"Conv2DLayer with a fq_output=None requires a weight "
                    f"of type QuantizedTensor, but got: {repr(weight)}"
                )
            if not issubclass(weight.layout_type, TensorScaledLayout):
                raise AssertionError(
                    f"Conv2DLayer with a fq_output=None requires a QuantizedTensor "
                    f"weight with TensorScaledLayout, but got: {weight.layout_type}"
                )

    def _validate_native_quant_mode(self):
        q_input = self.q_input
        q_output = self.q_output
        dq_output = self.dq_output

        if q_output is None and dq_output is None:
            raise AssertionError(
                f"One of q_output or dq_output must be specified in native "
                f"quantized mode for Conv2DLayer"
            )

        if q_output is not None and dq_output is not None:
            raise AssertionError(
                f"Only one of q_output or dq_output can be specified for a "
                f"Conv2DLayer but got both."
            )
