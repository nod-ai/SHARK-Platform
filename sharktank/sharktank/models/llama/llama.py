# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...layers import *
from ...types import *
from ...utils.create_cache import *
from ... import ops

__all__ = [
    "PagedLlamaModelV1",
]

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

    In the case of tensor sharding (config.tensor_parallelism_size > 1) the model's KV
    cache head dimension is sharded.
    The number of KV cache heads must be divisible by the parallelism size.
    With this sharding approach the KV cache is not replicated across devices.
    The cache is split across the devices while the indexing logic/computation is
    replicated.
    All other arguments aside from the cache state are replicated.
    After the attention we all-reduce.
    The the first fully connected layer is split along the parallel dimension.
    This drives that the reduction dimension is split for the second FC layer.
    We return the unreduced tensor. The user is free to reduce it to obtain the
    unsharded result or chain it with other tensor-parallel operations.
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
            fake_quant=config.fake_quant,
        )
        self.config = config
        self.hp = hp
        self.cache = create_kv_cache(self.config)
        self.activation_dtype = config.activation_dtype
        self.use_hf = config.use_hf
        self.attention_kernel = config.attention_kernel

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
                tensor_parallelism_size=config.tensor_parallelism_size,
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
                    attention_kernel=self.attention_kernel,
                    fake_quant=self.fake_quant,
                )
                for n in range(hp.block_count)
            ]
        )

    def prefill(
        self,
        # [bs, batch_seq_len]
        tokens: Union[torch.Tensor, ReplicatedTensor],
        *,
        # [1, 1, batch_seq_len, batch_seq_len]
        attention_mask: Union[torch.Tensor, ReplicatedTensor],
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: Union[torch.Tensor, ReplicatedTensor],
        cache_state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
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
        tokens: Union[torch.Tensor, ReplicatedTensor],
        *,
        # [bs, 1, 1, batch_seq_len]
        attention_mask: Union[torch.Tensor, ReplicatedTensor],
        # [bs] of starting positions
        start_positions: Union[torch.Tensor, ReplicatedTensor],
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: Union[torch.Tensor, ReplicatedTensor],
        cache_state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
    ):
        assert len(tokens.shape) == 2
        assert len(attention_mask.shape) == 4
        assert len(start_positions.shape) == 1
        assert len(seq_block_ids.shape) == 2
        assert tokens.shape[0] == attention_mask.shape[0]
        assert tokens.shape[0] == start_positions.shape[0]
        assert tokens.shape[0] == seq_block_ids.shape[0]
        assert tokens.shape[1] == 1
        assert attention_mask.shape[1] == 1 and attention_mask.shape[2] == 1
        assert (
            attention_mask.shape[3]
            == seq_block_ids.shape[1] * self.config.block_seq_stride
        )
        self._assert_device(tokens)
        self._assert_device(attention_mask, dtype=self.activation_dtype)
        self._assert_device(start_positions)
        self._assert_device(*cache_state, dtype=self.activation_dtype)

        if self.config.tensor_parallelism_size > 1:
            if not isinstance(tokens, ReplicatedTensor):
                tokens = ops.replicate(
                    tokens, count=self.config.tensor_parallelism_size
                )
            if not isinstance(attention_mask, ReplicatedTensor):
                attention_mask = ops.replicate(
                    attention_mask, count=self.config.tensor_parallelism_size
                )
            if not isinstance(start_positions, ReplicatedTensor):
                start_positions = ops.replicate(
                    start_positions, count=self.config.tensor_parallelism_size
                )
            if not isinstance(seq_block_ids, ReplicatedTensor):
                seq_block_ids = ops.replicate(
                    seq_block_ids, count=self.config.tensor_parallelism_size
                )
            # If the user provided unsharded arguments they probably want
            # an unsharded result as well.
            unshard_result = True
        else:
            unshard_result = False

        bs, _ = tokens.shape
        # Precompute a position based mask for computing rope embeddings
        # as it is the same for all blocks.
        embedding_batch_mask = self.attention_embedding.compute_batch_mask(
            start_positions, batch_seq_len=1
        )
        self.trace_tensor("llama.embedding_batch_mask", embedding_batch_mask)

        # Allocate per-block temporary K/V tensors. These temporaries hold
        # one block's K/V state for the maximum context length.
        if self.config.tensor_parallelism_size == 1:
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
        else:
            shard_size = [
                bs,
                self.context_length,
                self.hp.attention_head_count_kv // self.config.tensor_parallelism_size,
                self.hp.attn_head_dim,
            ]
            xk_temp_shard = [
                torch.empty(
                    shard_size, dtype=self.config.activation_dtype, device=self.device
                )
                for _ in range(self.config.tensor_parallelism_size)
            ]
            xv_temp_shard = [
                torch.empty(
                    shard_size, dtype=self.config.activation_dtype, device=self.device
                )
                for _ in range(self.config.tensor_parallelism_size)
            ]
            xk_temp = SplitPrimitiveTensor(ts=xk_temp_shard, shard_dim=2)
            xv_temp = SplitPrimitiveTensor(ts=xv_temp_shard, shard_dim=2)

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
        attention_kernel: str = "decomposed",
        fake_quant: bool = True,
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
                attention_kernel=attention_kernel,
                fake_quant=fake_quant,
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
        h: Union[torch.Tensor, ReplicatedTensor],
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
