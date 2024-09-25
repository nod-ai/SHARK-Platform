# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch
import torch.nn.functional as F
from .. import ops

from .base import Theta, ThetaLayer
from .linear import LinearLayer

__all__ = [
    "FFN",
]


class FFN(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
    ):
        super().__init__(theta)

        self.add_module("ffn_gate", LinearLayer(theta("ffn_gate")))
        self.add_module("ffn_up", LinearLayer(theta("ffn_up")))
        self.add_module("ffn_down", LinearLayer(theta("ffn_down")))

    def forward(
        self,
        h: torch.Tensor,
    ):
        ffn_gate = ops.elementwise(F.silu, self.ffn_gate(h))
        ffn_up = self.ffn_up(h)
        ffn_down = self.ffn_down(ffn_gate * ffn_up)
        return ffn_down
