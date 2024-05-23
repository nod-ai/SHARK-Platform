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
        self.add_module("gate", LinearLayer(theta("gate")))

        # Add FFN norm
        self.add_module(
            "ffn_norm", RMSNormLayer(theta("ffn_norm"), epsilon=rms_epsilon)
        )

        # Add num_experts x FFN experts
        self.experts = nn.ModuleList([FFN(theta) for _ in range(self.num_experts)])

    def forward(
        self,
        h: torch.Tensor,
    ):
        ffn_input = self.ffn_norm(h)
        batch_size, sequence_length, feature_dim = ffn_input.shape
        ffn_input = ffn_input.view(-1, feature_dim)

        # Given a token, the router calculates the routing weights for all experts
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(ffn_input)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # Select topk experts from routing weights
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k_experts, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # Cast back to the input dtype
        routing_weights = routing_weights.to(ffn_input.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, feature_dim), dtype=ffn_input.dtype
        )

        # Create an expert mask by one hot encoding the selected topk experts
        # used to index which expert is to be invoked
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(
            2, 1, 0
        )

        # Iterate over all experts in the model and perform computation on each expert
        for expert_idx in range(self.num_experts):
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


# TODO: Needs edit, pausibly used by layers/causal_llm.py if necessary, according to HF. Unused here.
def load_balancing_loss_func(
    gate_logits: torch.Tensor,
    num_experts: torch.Tensor = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
        )

    routing_weights = F.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = F.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (
            batch_size * sequence_length
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand(
                (num_hidden_layers, batch_size, sequence_length, top_k, num_experts)
            )
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask, dim=0
        ) / torch.sum(expert_attention_mask, dim=0)

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask, dim=0
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts
