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
    "FFNMOE",
]


class FFNMOE(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        expert_idx: Optional[int] = None,
    ):

        super().__init__(theta)

        try:
            self.add_module(
                "ffn_gate",
                LinearLayer(
                    theta.tensor("ffn_gate_exps", "weight").as_torch()[expert_idx]
                ),
            )
            self.add_module(
                "ffn_up",
                LinearLayer(
                    theta.tensor("ffn_up_exps", "weight").as_torch()[expert_idx]
                ),
            )
            self.add_module(
                "ffn_down",
                LinearLayer(
                    theta.tensor("ffn_down_exps", "weight").as_torch()[expert_idx]
                ),
            )
        except:
            self.add_module("ffn_gate", LinearLayer(theta("ffn_gate", expert_idx)))
            self.add_module("ffn_up", LinearLayer(theta("ffn_up", expert_idx)))
            self.add_module("ffn_down", LinearLayer(theta("ffn_down", expert_idx)))

    def forward(
        self,
        h: torch.Tensor,
    ):
        ffn_gate = F.silu(self.ffn_gate(h))
        ffn_up = self.ffn_up(h)
        ffn_down = self.ffn_down(ffn_gate * ffn_up)
        return ffn_down
