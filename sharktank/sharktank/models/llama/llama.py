# Copyright 2024 Advanced Micro Devices, Inc
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
        self.hf = False
        self.config = config
        self.hp = hp
        self.cache = config.create_kv_cache()
        self.activation_dtype = config.activation_dtype
        self.use_hf = config.use_hf

        
        key = "token_embd"
        if key not in list(theta.keys):
            self.hf = True
            key = "model.embed_tokens"
        self.add_module(
            "token_embedding",
            TokenEmbeddingLayer(theta(key), dtype=config.activation_dtype),
        )
        self.add_module(
            "attention_embedding",
            RotaryEmbeddingLayer(
                rope_dimension_count=hp.rope_dimension_count,
                max_seqlen=hp.context_length,
                device=self.device,
                use_hf=self.use_hf,
                static_tables=config.static_tables,
            ),
        )
        key = "output_norm" if "output_norm" in list(theta.keys) else "model.norm"
        self.add_module(
            "output_norm",
            RMSNormLayer(
                theta(key), epsilon=self.hp.attention_layer_norm_rms_epsilon
            ),
        )
        print(theta.keys)
        key = "output" if "output" in list(theta.keys) else "lm_head"
        self.add_module("output_lm_head", LinearLayer(theta(key)))
        key = "blk" if "blk" in list(theta.keys) else "model.layers"
        self.attn_blocks = nn.ModuleList(
            [
                PagedLlamaAttentionBlock(
                    theta(key, n),
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


class PagedLlamaAttentionBlock(ThetaLayer):
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
            "attn_norm", RMSNormLayer(theta("attn_norm"), epsilon=rms_epsilon)
        )
        self.add_module("attn_q", LinearLayer(theta("attn_q")))
        self.add_module("attn_k", LinearLayer(theta("attn_k")))
        self.add_module("attn_v", LinearLayer(theta("attn_v")))
        self.add_module("attn_output", LinearLayer(theta("attn_output")))
        self.add_module(
            "ffn_norm", RMSNormLayer(theta("ffn_norm"), epsilon=rms_epsilon)
        )
        self.add_module("ffn_gate", LinearLayer(theta("ffn_gate")))
        self.add_module("ffn_up", LinearLayer(theta("ffn_up")))
        self.add_module("ffn_down", LinearLayer(theta("ffn_down")))
    ):  
        super().__init__(theta)
        if hf:
            #tensor = theta("self_attn.qkv.weight").tensor
            #tensor = tensor.reshape(head_count_kv, head_count // head_count_kv + 2, head_dim, head_dim * head_count)
            #print(tensor)
            self.add_module("attn_norm", RMSNormLayer(theta("input_layernorm"), epsilon=rms_epsilon))
            #self.add_module("attn_qkv", LinearLayer(theta("self_attn.qkv")))
            self.add_module("attn_q", LinearLayer(theta("self_attn.q_proj")))
            self.add_module("attn_k", LinearLayer(theta("self_attn.k_proj")))
            self.add_module("attn_v", LinearLayer(theta("self_attn.v_proj")))
            self.add_module("attn_output", LinearLayer(theta("self_attn.o_proj")))
            self.add_module("ffn_norm", RMSNormLayer(theta("post_attention_layernorm"), epsilon=rms_epsilon))
            self.add_module("ffn_gate", LinearLayer(theta("mlp.gate_proj")))
            self.add_module("ffn_up", LinearLayer(theta("mlp.up_proj")))
            self.add_module("ffn_down", LinearLayer(theta("mlp.down_proj")))
        else:
            self.add_module(
                "attn_norm", RMSNormLayer(theta("attn_norm"), epsilon=rms_epsilon)
            )
            self.add_module("attn_q", LinearLayer(theta("attn_q")))
            self.add_module("attn_k", LinearLayer(theta("attn_k")))
            self.add_module("attn_v", LinearLayer(theta("attn_v")))
            self.add_module("attn_output", LinearLayer(theta("attn_output")))
            self.add_module(
                "ffn_norm", RMSNormLayer(theta("ffn_norm"), epsilon=rms_epsilon)
            )
            self.add_module("ffn_gate", LinearLayer(theta("ffn_gate")))
            self.add_module("ffn_up", LinearLayer(theta("ffn_up")))
            self.add_module("ffn_down", LinearLayer(theta("ffn_down")))

        self.block_index = block_index
        self.cache = cache
        self.head_count = head_count
        self.head_dim = head_dim
        self.head_count_kv = head_count_kv
        self.use_hf = use_hf

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
        assert bool(start_index is not None) ^ bool(embedding_batch_mask is not None)

        x = self.attn_norm(h)

        bs, batch_seq_len, feature_dim = x.shape
        assert feature_dim == self.head_count * self.head_dim

        xq = self.attn_q(x)
        xk = self.attn_k(x)
        xv = self.attn_v(x)

        xq = xq.view(bs, batch_seq_len, self.head_count, self.head_dim)
        xk = xk.view(bs, batch_seq_len, self.head_count_kv, self.head_dim)
        xv = xv.view(bs, batch_seq_len, self.head_count_kv, self.head_dim)

        # Fast path to start_index based embedding lookup if available.
        # Falls back to a slower position based index lookup.
        if start_index is not None:
            xq, xk = embedding.forward(xq=xq, xk=xk, start_index=start_index)
        else:
            xq, xk = embedding.apply_batched_mask(
                xq=xq, xk=xk, mask=embedding_batch_mask
            )

        # Full sequence length.
        kv_seq_len = seq_block_ids.shape[1] * self.cache.block_seq_stride

        if self.cache.is_paged:
            xk, xv = self.transact_cache_paged(
                xk_cache_update=xk,
                xv_cache_update=xv,
                seq_block_ids=seq_block_ids,
                kv_seq_len=kv_seq_len,
                start_positions=start_positions,
                cache_state=cache_state,
                xk_temp=xk_temp,
                xv_temp=xv_temp,
            )
        elif self.cache.is_direct:
            xk, xv = self.transact_cache_direct(
                xk_cache_update=xk,
                xv_cache_update=xv,
                start_positions=start_positions,
                kv_seq_len=kv_seq_len,
                cache_state=cache_state,
            )
        else:
            raise NotImplementedError(f"Unsupported KV cache type: {type(self.cache)}")

        # Expand kv heads for GQA.
        gqa_n_rep = self.head_count // self.head_count_kv
        assert gqa_n_rep > 0
        if gqa_n_rep > 1:

            def repeat_kv(x: torch.Tensor) -> torch.Tensor:
                bs, slen, n_kv_heads, head_dim = x.shape
                return (
                    x.unsqueeze(-2)
                    .expand(bs, slen, n_kv_heads, gqa_n_rep, head_dim)
                    .reshape(bs, slen, n_kv_heads * gqa_n_rep, head_dim)
                )

            xk = repeat_kv(xk)
            xv = repeat_kv(xv)

        # Transpose into [bs, heads, sl, dim]
        xq = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)

        # Flash attention.
        attn_weights = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        self.assert_not_nan(attn_weights)

        # Apply attention mask.
        self.trace_tensor("attn_weights", attn_weights, values=False)
        if attention_mask is not None:
            # self.trace_tensor("attn_mask", attention_mask)
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(xq)
        attn_output = torch.matmul(attn_weights, values)  # (bs, heads, slen, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bs, batch_seq_len, -1)

        # Project.
        attn_output = self.attn_output(attn_output)

        # Remainder of the block.
        h = h + attn_output

        # Feed forward network.
        ffn_input = self.ffn_norm(h)
        ffn_gate = F.silu(self.ffn_gate(ffn_input))
        ffn_up = self.ffn_up(ffn_input)
        ffn_down = self.ffn_down(ffn_gate * ffn_up)
        final_output = h + ffn_down

        return final_output

    def transact_cache_direct(
        self,
        *,
        cache_state: list[torch.Tensor],
        xk_cache_update: torch.Tensor,
        xv_cache_update: torch.Tensor,
        kv_seq_len: int,
        start_positions: Optional[torch.Tensor] = None,
    ):
        bs, batch_seq_len, _, _ = xk_cache_update.shape
        cache_k = cache_state[self.block_index * 2]
        cache_v = cache_state[self.block_index * 2 + 1]

        if start_positions is None:
            # Prefill. Write the entire cache.
            cache_k[:, :batch_seq_len] = xk_cache_update
            cache_v[:, :batch_seq_len] = xv_cache_update
            return xk_cache_update, xv_cache_update
        else:
            # Decode. Write a single timestep.
            # TODO: This needs to be reworked with index ops.
            assert xk_cache_update.shape[1] == 1
            assert xv_cache_update.shape[1] == 1
            max_start_pos = 0
            for row_index in range(bs):
                row_start_pos = start_positions[row_index].item()
                max_start_pos = max(row_start_pos, max_start_pos)
                cache_k[row_index, row_start_pos] = xk_cache_update[row_index, 0]
                cache_v[row_index, row_start_pos] = xv_cache_update[row_index, 0]
            return cache_k[:, :kv_seq_len], cache_v[:, :kv_seq_len]

    def transact_cache_paged(
        self,
        *,
        xk_cache_update: torch.Tensor,
        xv_cache_update: torch.Tensor,
        cache_state: list[torch.Tensor],
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        kv_seq_len: int,
        start_positions: Optional[torch.Tensor] = None,
        xk_temp: Optional[torch.Tensor] = None,
        xv_temp: Optional[torch.Tensor] = None,
    ):
        cache = self.cache.paged
        # Manage the cache.
        if start_positions is None:
            # Prefill: Write the entire cache.
            cache.write(
                cache_state,
                cache_partitions=[xk_cache_update, xv_cache_update],
                transformer_block_index=self.block_index,
                page_ids=seq_block_ids,
            )
            return xk_cache_update, xv_cache_update
        else:
            # Decode at ragged start positions.
            # We need to initialize/read the K/V from the cache for the whole
            # sequence. Note that at this point, it is possible to fork and
            # use a memory efficient attention kernel that can do indirect
            # reads, skipping this materialization. This path is taken for
            # a decode step.
            assert xk_temp is not None and xv_temp is not None
            assert xk_cache_update.shape[1] == 1
            assert xv_cache_update.shape[1] == 1
            assert kv_seq_len == seq_block_ids.shape[1] * cache.block_seq_stride

            # Write our one updated cache row into the cache.
            cache.write_timestep(
                cache_state,
                cache_partitions=[
                    xk_cache_update,
                    xv_cache_update,
                ],
                transformer_block_index=self.block_index,
                seq_positions=start_positions,
                page_ids=seq_block_ids,
            )

            # Restore from the cache.
            cache.read(
                cache_state,
                read_into_partitions=[
                    xk_temp[:, 0:kv_seq_len, ...],
                    xv_temp[:, 0:kv_seq_len, ...],
                ],
                transformer_block_index=self.block_index,
                page_ids=seq_block_ids,
            )

            # For computation, we create a subview of the xk/xv tensors to have
            # a sequence length covering the blocked size. This must include
            # the newly added row (the caller is responsible for ensuring that
            # every block has at least one row left). We'll compute on this
            # ragged view and use an appropriate mask.
            xk = xk_temp[:, 0:kv_seq_len, ...]
            xv = xv_temp[:, 0:kv_seq_len, ...]
            return xk, xv
