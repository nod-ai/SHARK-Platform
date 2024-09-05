# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...layers import *
from ...types import *
from ... import ops

__all__ = [
    "LlamaModelConfig",
    "PagedLlamaModelV1",
]

################################################################################
# Config
################################################################################


@dataclass
class LlamaModelConfig:
    hp: configs.LlamaHParams

    # Block sequence stride for a paged KV cache. This must divide evenly
    # into the context length.
    block_seq_stride: int = 16

    # Either "paged" or "direct".
    kv_cache_type: str = "paged"

    # The device on which to place intermediate state.
    device: Optional[torch.device] = None

    # Dtype to use for general FP activations not otherwise configured.
    activation_dtype: torch.dtype = torch.float16

    # Dtype to use for attention.
    attention_dtype: torch.dtype = torch.float16

    # Indicates if running with HuggingFace implementation and ensures
    # numerical equivalency to HuggingFace's LLaMa if true (by modifying
    # rotary embedding).
    use_hf: bool = False

    # If true, then the model may pre-initialize certain tables during
    # init. This can be better for eager execution but when capturing a program,
    # it is often better to preserve the calculation explicitly and rely on
    # the compiler to transform it to an initialization time step. This can
    # be the difference of many gigabytes of static data being embedded in
    # the program and not.
    static_tables: bool = True

    def create_kv_cache(self) -> BaseKVCache:
        hp = self.hp
        if self.kv_cache_type == "direct":
            return DirectKVCache(
                block_seq_stride=self.block_seq_stride,
                transformer_block_count=hp.block_count,
                attn_head_count=hp.attention_head_count_kv,
                attn_head_dim=hp.attn_head_dim,
                seq_length=hp.context_length,
                device=self.device,
                dtype=self.attention_dtype,
            )
        elif self.kv_cache_type == "paged":
            return PagedKVCache(
                transformer_block_count=hp.block_count,
                attn_head_count=hp.attention_head_count_kv,
                attn_head_dim=hp.attn_head_dim,
                cache_partition_count=2,  # One for each of K/V.
                block_seq_stride=self.block_seq_stride,
                device=self.device,
                dtype=self.attention_dtype,
            )
        else:
            raise NotImplementedError(f"kv_cache_type = {self.kv_cache_type}")


################################################################################
# Models
################################################################################


class PagedLlamaModelV1(BaseCausalLMModel):
    """LlamaModel with a paged KV cache and supporting variable sequence
    length batched inference.

    As both the caching and batching setup is complicated, this model variant
    is modular, intending to be instantiated and used in an overall assembly
    vs trying to providing one-stop methods that do everything.

    The inference procedure is typically:

    1. Initialize the PagedKVCache state tensors.
    2. Generate an input mask given a vector of sequence lengths.
    3. Generate an attention mask from the input mask.
    4. Allocate a block mapping table.
    5. Invoke prefill() with a batch of sequences.
    6. Extract tokens from batched logits.
    7. Iteratively invoke decode() for as long as there are sequences needing
       to be serviced.

    Various samplers and schedulers can be interleaved throughout.
    """

    def __init__(self, theta: Theta, config: LlamaModelConfig):
        hp = config.hp
        super().__init__(
            theta,
            context_length=config.hp.context_length,
            static_tables=config.static_tables,
            device=config.device,
            activation_dtype=config.activation_dtype,
            attention_dtype=config.attention_dtype,
        )
        self.config = config
        self.hp = hp
        self.cache = config.create_kv_cache()
        self.activation_dtype = config.activation_dtype
        self.use_hf = config.use_hf

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
                device=self.device,
                use_hf=self.use_hf,
                static_tables=config.static_tables,
            ),
        )
        self.add_module(
            "output_norm",
            RMSNormLayer(
                theta("output_norm"), epsilon=self.hp.attention_layer_norm_rms_epsilon
            ),
        )
        self.add_module("output_lm_head", LinearLayer(theta("output")))

        self.attn_blocks = nn.ModuleList(
            [
                AttentionFFNBlock(
                    theta("blk", n),
                    block_index=n,
                    cache=self.cache,
                    head_count=hp.attention_head_count,
                    head_dim=hp.attn_head_dim,
                    head_count_kv=hp.attention_head_count_kv,
                    rms_epsilon=hp.attention_layer_norm_rms_epsilon,
                    use_hf=self.use_hf,
                )
                for n in range(hp.block_count)
            ]
        )

    def prefill(
        self,
        # [bs, batch_seq_len]
        tokens: torch.Tensor,
        *,
        # [1, 1, batch_seq_len, batch_seq_len]
        attention_mask: torch.Tensor,
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        cache_state: list[torch.Tensor],
    ):
        self._assert_device(tokens)
        self._assert_device(attention_mask, dtype=self.activation_dtype)
        self._assert_device(seq_block_ids)
        self._assert_device(*cache_state, dtype=self.activation_dtype)
        h = self.token_embedding(tokens)
        self.trace_tensor("llama.token_embedding", h)

        # Iterate over attention blocks.
        for block_idx, block in enumerate(self.attn_blocks):
            if block_idx == 0:
                self.trace_tensor(f"llama.attn_block.{block_idx}.input", h)
            h = block(
                h,
                embedding=self.attention_embedding,
                start_index=0,
                attention_mask=attention_mask,
                cache_state=cache_state,
                seq_block_ids=seq_block_ids,
            )
            self.trace_tensor(f"llama.attn_block.{block_idx}.output", h)

        h = self.output_norm(h)
        logits = self.output_lm_head(h)
        return logits

    def decode(
        self,
        # [bs, 1]
        tokens: torch.Tensor,
        *,
        # [bs, 1, 1, batch_seq_len]
        attention_mask: torch.Tensor,
        # [bs] of starting positions
        start_positions: torch.Tensor,
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        cache_state: list[torch.Tensor],
    ):
        self._assert_device(tokens)
        self._assert_device(attention_mask, dtype=self.activation_dtype)
        self._assert_device(start_positions)
        self._assert_device(*cache_state, dtype=self.activation_dtype)
        bs, _ = tokens.shape
        # Precompute a position based mask for computing rope embeddings
        # as it is the same for all blocks.
        embedding_batch_mask = self.attention_embedding.compute_batch_mask(
            start_positions, batch_seq_len=1
        )
        self.trace_tensor("llama.embedding_batch_mask", embedding_batch_mask)

        # Allocate per-block temporary K/V tensors. These temporaries hold
        # one block's K/V state for the maximum context length.
        xk_temp = torch.empty(
            [
                bs,
                self.context_length,
                self.hp.attention_head_count_kv,
                self.hp.attn_head_dim,
            ],
            dtype=self.config.activation_dtype,
            device=self.device,
        )
        xv_temp = torch.empty(
            [
                bs,
                self.context_length,
                self.hp.attention_head_count_kv,
                self.hp.attn_head_dim,
            ],
            dtype=self.config.activation_dtype,
            device=self.device,
        )

        h = self.token_embedding(tokens)
        self.trace_tensor("llama.token_embedding", h)

        # Iterate over attention blocks.
        for block_idx, block in enumerate(self.attn_blocks):
            if block_idx == 0:
                self.trace_tensor(f"llama.attn_block.{block_idx}.input", h)
            h = block(
                h,
                start_positions=start_positions,
                embedding=self.attention_embedding,
                embedding_batch_mask=embedding_batch_mask,
                attention_mask=attention_mask,
                cache_state=cache_state,
                seq_block_ids=seq_block_ids,
                xk_temp=xk_temp,
                xv_temp=xv_temp,
            )
            self.trace_tensor(f"llama.attn_block.{block_idx}.output", h)

        h = self.output_norm(h)
        logits = self.output_lm_head(h)
        return logits


################################################################################
# Layers
################################################################################


class AttentionFFNBlock(ThetaLayer):
    """Implements a self attention layer in the style of Llama using a
    paged cache."""

    def __init__(
        self,
        theta: Theta,
        *,
        block_index: int,
        cache: PagedKVCache,
        head_count: int,
        head_dim: int,
        head_count_kv: int,
        rms_epsilon: float,
        use_hf: bool = False,
    ):
        super().__init__(theta)
        self.add_module(
            "attn",
            PagedLlamaAttentionBlock(
                theta=theta,
                block_index=block_index,
                cache=cache,
                head_count=head_count,
                head_dim=head_dim,
                head_count_kv=head_count_kv,
                rms_epsilon=rms_epsilon,
                use_hf=use_hf,
            ),
        )
        self.add_module(
            "ffn",
            FFN(
                theta=theta,
            ),
        )
        self.add_module(
            "ffn_norm", RMSNormLayer(theta("ffn_norm"), epsilon=rms_epsilon)
        )

    def forward(
        self,
        h: torch.Tensor,
        *,
        embedding: RotaryEmbeddingLayer,
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        start_index: Optional[int] = None,
        start_positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        embedding_batch_mask: Optional[torch.Tensor] = None,
        cache_state: list[torch.Tensor] = None,
        xk_temp: Optional[torch.Tensor] = None,
        xv_temp: Optional[torch.Tensor] = None,
    ):
        h = self.attn(
            h,
            embedding=embedding,
            seq_block_ids=seq_block_ids,
            start_index=start_index,
            start_positions=start_positions,
            attention_mask=attention_mask,
            embedding_batch_mask=embedding_batch_mask,
            cache_state=cache_state,
            xk_temp=xk_temp,
            xv_temp=xv_temp,
        )
        # Feed forward network.
        ffn_input = self.ffn_norm(h)
        ffn_down = self.ffn(ffn_input)
        final_output = h + ffn_down

        return final_output
