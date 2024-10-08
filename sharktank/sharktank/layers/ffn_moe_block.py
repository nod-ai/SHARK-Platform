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
from ..ops import einsum_2args, matmul

__all__ = [
    "FFNMOE",
    "PreGatherFFNMOE",
]


class PreGatherFFNMOE(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        use_grok: bool = False,
    ):

        super().__init__(theta)
        self.use_grok = use_grok

        self.ffn_gate = theta.tensor("ffn_gate_exps", "weight")
        self.ffn_up = theta.tensor("ffn_up_exps", "weight")
        self.ffn_down = theta.tensor("ffn_down_exps", "weight")

    def pre_matmul_gather(self, inputs, weights, experts):
        inputs = inputs[:, :]
        weights = weights[experts, :, :]
        result = einsum_2args(inputs, weights, "mk,menk->men")
        return result

    def bigger_mmg(self, inputs, weights, experts):
        inputs = inputs[:, :]
        weights = weights[experts, :, :]
        result = einsum_2args(inputs, weights, "mek,menk->men")
        return result

    def one_hot_matmul(self, inputs, weights, experts):
        # batch dims: none, lhs pdims: m, lhs rdims: k, rhs pdims: bn, rhs rdims: k
        matmul = einsum_2args(inputs, weights, "mk,bnk->bmn")
        # Post mix the experts
        oh = (
            torch.nn.functional.one_hot(experts.reshape(-1), num_classes=8)
            .transpose(0, 1)
            .to(torch.float32)
        )
        output = einsum_2args(oh, matmul, "bm,bmn->mn")
        # batch dims: m, lhs pdims: none, lhs rdims: b, rhs pdims: n, rhs rdims: b
        # inputs_shape = inputs.shape
        # inputs = inputs.view(inputs_shape[0], 1, inputs_shape[1])
        # inputs = inputs.transpose(0, 2)

        # weights_shape = weights.shape
        # weights = weights.permute(1, 2, 0)

        # result = matmul(inputs, weights)
        # result = result.view(weights_shape[1], weights_shape[2])
        return output

    def forward(
        self,
        h: torch.Tensor,
        experts: torch.Tensor,
        expert_gate: torch.Tensor,
    ):
        if self.use_grok:
            ffn_gate = F.gelu(self.pre_matmul_gather(h, self.ffn_gate, experts))
        else:
            ffn_gate = F.silu(self.pre_matmul_gather(h, self.ffn_gate, experts))

        ffn_up = self.pre_matmul_gather(h, self.ffn_up, experts)
        ffn_down = self.bigger_mmg(ffn_gate * ffn_up, self.ffn_down, experts)
        ffn_down = einsum_2args(expert_gate, ffn_down, "me,men->men")
        return torch.sum(ffn_down, dim=1)


class FFNMOE(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        expert_idx: Optional[int] = None,
    ):

        super().__init__(theta)

        if theta.optional_tensor("ffn_gate_exps") is not None:
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
    # fetches the block_idx from merged_tensor_name. e.g. blk.0.ffn_gate_exps.weight
    expert_layer_name = (
        f"blk.{merged_tensor.name.split('.')[1]}.{layer_name}.{expert_idx}.weight"
    )
    expert_tensor = DefaultPrimitiveTensor(
        name=expert_layer_name, data=merged_tensor.as_torch()[expert_idx]
    )
    return expert_tensor
