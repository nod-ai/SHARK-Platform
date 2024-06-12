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
        self.q_input: Optional[QuantizerTensor] = theta.optional_tensor("q_input")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        q_input = self.q_input
        weight = self.weight
        bias = self.bias

        if self.premul_input is not None:
            x = ops.elementwise(torch.mul, x, self.premul_input)

        if q_input is not None:
            x = q_input.quantize(x)

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

        # Unconditionally dequantize.
        # TODO: Support a q_output specifier that signals the layer to let
        # the QuantizedTensor escape.
        if isinstance(y, QuantizedTensor):
            y = y.unpack().dequant()
        return y
