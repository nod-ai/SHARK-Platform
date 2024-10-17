# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Theta, ThetaLayer
from .linear import LinearLayer
from .norm import RMSNormLayer
from .ffn_moe_block import FFNMOE, PreGatherFFNMOE

__all__ = [
    "MoeBlock",
]


class MoeBlock(ThetaLayer):
    """
    This implementation considers MoE operations as block-sparse
    operations to support imbalanced token assignments to experts.
    This enables the MoE to operate at a faster rate and in full capacity without any dropped tokens
    (or reduced performance).
    """

    def __init__(
        self,
        theta: Theta,
        expert_count: int,
        expert_used_count: int,
        rms_epsilon: float,
        moe_activation=F.silu,
    ):
        super().__init__(theta)

        self.expert_count = expert_count
        self.expert_used_count = expert_used_count

        # Add router gate
        self.add_module("ffn_gate_inp", LinearLayer(theta("ffn_gate_inp")))

        # Add FFN norm
        self.add_module(
            "ffn_norm", RMSNormLayer(theta("ffn_norm"), epsilon=rms_epsilon)
        )

        # Add optional FFN output norm layer
        if theta.optional_tensor("layer_output_norm") is not None:
            self.add_module(
                "layer_output_norm",
                RMSNormLayer(theta("layer_output_norm"), epsilon=rms_epsilon),
            )
        else:
            self.add_module("layer_output_norm", torch.nn.Identity())

        # Add expert_count x FFN
        self.experts = PreGatherFFNMOE(theta, activation=moe_activation)

    def forward(
        self,
        h: torch.Tensor,
    ):
        ffn_input = self.ffn_norm(h)
        batch_size, sequence_length, feature_dim = ffn_input.shape
        ffn_input = ffn_input.view(-1, feature_dim)

        # For each token, the router calculates the router weights for all experts
        # router_logits: (batch_size * sequence_length, expert_count)
        router_logits = self.ffn_gate_inp(ffn_input)
        router_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # Select top k experts from router weights
        expert_gate, top_k_experts = torch.topk(
            router_weights, self.expert_used_count, dim=-1
        )

        expert_gate /= expert_gate.sum(dim=-1, keepdim=True)
        expert_gate = expert_gate.to(ffn_input.dtype)

        moe_output = self.experts(ffn_input, top_k_experts, expert_gate)
        moe_output = moe_output.reshape(batch_size, sequence_length, feature_dim)

        moe_output = self.layer_output_norm(moe_output)

        return h + moe_output
