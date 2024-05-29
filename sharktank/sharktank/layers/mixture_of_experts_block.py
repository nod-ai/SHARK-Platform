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
from .ffn_block import FFN

__all__ = [
    "SparseMoeBlock",
]


class SparseMoeBlock(ThetaLayer):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(
        self,
        theta: Theta,
        num_experts: int,
        top_k_experts: int,
        rms_epsilon: float,
    ):
        super().__init__(theta)

        # Add gating
        self.add_module("ffn_gate_inp", LinearLayer(theta("ffn_gate_inp")))

        # Add FFN norm
        self.add_module(
            "ffn_norm", RMSNormLayer(theta("ffn_norm"), epsilon=rms_epsilon)
        )

        # Add num_experts x FFN experts
        self.experts = nn.ModuleList(
            [FFN(theta, expert_idx=i) for i in range(num_experts)]
        )

    def forward(
        self,
        h: torch.Tensor,
    ):
        ffn_input = self.ffn_norm(h)
        batch_size, sequence_length, feature_dim = ffn_input.shape
        ffn_input = ffn_input.view(-1, feature_dim)

        # Given a token, the router calculates the routing weights for all experts
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.ffn_gate_inp(ffn_input)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # Select topk experts from routing weights
        routing_weights, selected_experts = torch.topk(
            routing_weights, top_k_experts, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # Cast back to the input dtype
        routing_weights = routing_weights.to(ffn_input.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, feature_dim), dtype=ffn_input.dtype
        )

        # Create an expert mask by one hot encoding the selected topk experts
        # used to index which expert is to be invoked
        expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(
            2, 1, 0
        )

        # Iterate over all experts in the model and perform computation on each expert
        for expert_idx in range(num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = ffn_input[None, top_x].reshape(-1, feature_dim)
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(ffn_input.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, feature_dim
        )
        return h + final_hidden_states, router_logits
