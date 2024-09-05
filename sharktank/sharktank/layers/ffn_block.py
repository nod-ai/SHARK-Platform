# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch
import torch.nn.functional as F

from .base import Theta, ThetaLayer
from .linear import LinearLayer

__all__ = [
    "FFN",
]


class FFN(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        expert_idx: Optional[int] = None,
    ):
        super().__init__(theta)
        ffn_g = "ffn_gate"
        ffn_u = "ffn_up"
        ffn_d = "ffn_down"
        if expert_idx is not None:
            ffn_g = f"ffn_gate.{expert_idx}"
            ffn_u = f"ffn_up.{expert_idx}"
            ffn_d = f"ffn_down.{expert_idx}"
        self.add_module("ffn_gate", LinearLayer(theta(ffn_g)))
        self.add_module("ffn_up", LinearLayer(theta(ffn_u)))
        self.add_module("ffn_down", LinearLayer(theta(ffn_d)))

    def forward(
        self,
        h: torch.Tensor,
    ):
        ffn_gate = F.silu(self.ffn_gate(h))
        ffn_up = self.ffn_up(h)
        ffn_down = self.ffn_down(ffn_gate * ffn_up)
        return ffn_down
