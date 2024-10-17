# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass

import torch
import torch.nn as nn

from ...layers import *
from ...types import Theta

__all__ = [
    "RefLlamaModelConfig",
    "DirectCacheMixtralModelV1",
]


################################################################################
# Config
################################################################################


@dataclass
class RefLlamaModelConfig:
    hp: configs.LlamaHParams

    # Dtype to use for general FP activations not otherwise configured.
    activation_dtype: torch.dtype = torch.float16


################################################################################
# Models
################################################################################


class DirectCacheMixtralModelV1(ThetaLayer):
    """Simple Mixtral Model with a direct lookup KV cache for batch-1 inference."""

    def __init__(self, theta: Theta, config: RefLlamaModelConfig):
        super().__init__(theta)
        hp = config.hp
        self.config = config
        self.hp = hp
        self.activation_dtype = config.activation_dtype
        self.add_module(
            "token_embedding",
            TokenEmbeddingLayer(theta("token_embd"), dtype=config.activation_dtype),
        )
        self.add_module(
            "attention_embedding",
            RotaryEmbeddingLayer(
                rope_dimension_count=hp.rope_dimension_count,
                rope_freq_base=hp.rope_freq_base,
                max_seqlen=hp.context_length,
            ),
        )
        self.add_module(
            "output_norm",
            RMSNormLayer(
                theta("output_norm"), epsilon=self.hp.attention_layer_norm_rms_epsilon
            ),
        )
        self.add_module("output_lm_head", LinearLayer(theta("output")))

        self.attn_blocks = nn.ModuleList()
        self.moe_blocks = nn.ModuleList()

        for n in range(hp.block_count):
            self.attn_blocks.append(
                LlamaAttentionBlock(
                    theta("blk", n),
                    embedding=self.attention_embedding,
                    head_count=hp.attention_head_count,
                    head_dim=hp.rope_dimension_count,
                    head_count_kv=hp.attention_head_count_kv,
                    rms_epsilon=hp.attention_layer_norm_rms_epsilon,
                )
            )
            self.moe_blocks.append(
                MoeBlock(
                    theta("blk", n),
                    expert_count=hp.expert_count,
                    expert_used_count=hp.expert_used_count,
                    rms_epsilon=hp.attention_layer_norm_rms_epsilon,
                )
            )

    def create_cache(self, bs: int) -> list[torch.Tensor]:
        return [
            torch.empty(
                (
                    bs,
                    self.hp.context_length,
                    self.hp.attention_head_count,
                    self.hp.rope_dimension_count,
                ),
                dtype=self.activation_dtype,
            )
            for _ in range(self.hp.block_count * 2)
        ]

    def forward(
        self,
        tokens: torch.Tensor,
        start_index: int,
        *,
        return_logits: bool = False,
        local_kv_cache: list[torch.Tensor],
    ):
        bs, sl = tokens.shape
        h = self.token_embedding(tokens)
        dtype = h.dtype
        self.trace_tensor("mixtral.token_embedding", h)

        # Compute attention mask.
        attention_mask = None
        if sl > 1:
            # Use the smallest value like HF as opposed to -inf like original.
            # A little bit easier for some systems.
            attention_mask = torch.full(
                (1, 1, sl, sl), torch.finfo(dtype).min, dtype=dtype
            )
            attention_mask = torch.triu(
                attention_mask, diagonal=start_index + 1
            ).type_as(h)

        # block_count = (total no. of blocks)/2, excluding MoE blocks
        block_count = len(self.attn_blocks) // 2
        # print('local_kv_cache, #attn_blocks', len(local_kv_cache), block_count)
        # Iterate over attention + MoE blocks.
        for block_idx, (attn_block, moe_block) in enumerate(
            zip(self.attn_blocks, self.moe_blocks)
        ):
            # print("block_idx, block", block_idx, block)
            if block_idx == 0:
                self.trace_tensor(f"mixtral.attn_block.{block_idx}.input", h)

            attn_block_idx = block_idx // 2
            block_cache_k = local_kv_cache[attn_block_idx]
            block_cache_v = local_kv_cache[block_count + attn_block_idx]
            h = attn_block(
                h,
                cache_k=block_cache_k,
                cache_v=block_cache_v,
                start_index=start_index,
                attention_mask=attention_mask,
            )
            self.trace_tensor(f"mixtral.attn_block.{block_idx}.output", h)

            h = attn_block(
                h,
            )
            self.trace_tensor(f"mixtral.moe_block.{block_idx}.output", h)

        h = self.output_norm(h)
        logits = self.output_lm_head(h)

        if return_logits:
            return h
        else:
            last_step = logits[:, -1, :]
            token = torch.argmax(last_step, keepdim=True, dim=1)
            return token.to(tokens.dtype)
