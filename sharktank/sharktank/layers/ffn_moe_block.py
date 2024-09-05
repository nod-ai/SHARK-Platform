# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch
import torch.nn.functional as F

from .base import ThetaLayer
from .linear import LinearLayer
from ..types import Theta, DefaultPrimitiveTensor

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

        if theta.optional_tensor("ffn_gate_exps") is not None:
            ''' 
            Expands a single merged expert tensor to individual expert tensors 
            Eg: Converts blk.0.ffn_gate_exps.weight to blk.0.ffn_gate.0.weight, blk.0.ffn_gate.1.weight, etc.

            '''
          
            merged_tensor = theta.tensor("ffn_gate_exps", "weight")

            expert_tensor = extract_ffn_layer(
                merged_tensor=merged_tensor,
                layer_name="ffn_gate",
                expert_idx=expert_idx,
            )

            self.add_module("ffn_gate", LinearLayer(Theta({"weight": expert_tensor})))

            merged_tensor = theta.tensor("ffn_up_exps", "weight")

            expert_tensor = extract_ffn_layer(
                merged_tensor=merged_tensor, layer_name="ffn_up", expert_idx=expert_idx
            )

            self.add_module("ffn_up", LinearLayer(Theta({"weight": expert_tensor})))

            merged_tensor = theta.tensor("ffn_down_exps", "weight")

            expert_tensor = extract_ffn_layer(
                merged_tensor=merged_tensor,
                layer_name="ffn_down",
                expert_idx=expert_idx,
            )

            self.add_module("ffn_down", LinearLayer(Theta({"weight": expert_tensor})))

        else:
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


def extract_ffn_layer(
    merged_tensor: DefaultPrimitiveTensor, layer_name: str, expert_idx: int
):
    '''
    Given a merged expert tensor and an expert_idx, extracts the respective expert tensor 
    and constructs a DefaultPrimitiveTensor with the relevant expert layer name
    '''

    expert_layer_name = (
        f"blk.{merged_tensor.name.split('.')[1]}.{layer_name}.{expert_idx}.weight"
    )
    expert_tensor = DefaultPrimitiveTensor(
        name=expert_layer_name, data=merged_tensor.as_torch()[expert_idx]
    )
    return expert_tensor
