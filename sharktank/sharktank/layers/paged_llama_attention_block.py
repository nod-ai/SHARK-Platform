# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import math

import torch
import torch.nn.functional as F

from .base import Theta, ThetaLayer
from .linear import LinearLayer
from .norm import RMSNormLayer
from .rotary_embedding import RotaryEmbeddingLayer
from .kv_cache import PagedKVCache
from .. import ops

__all__ = [
    "PagedLlamaAttentionBlock",
]


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
        attention_scale: Optional[float] = None,
        softcap: Optional[float] = None,
    ):
        super().__init__(theta)

        self.block_index = block_index
        self.cache = cache
        self.head_count = head_count
        self.head_dim = head_dim
        self.head_count_kv = head_count_kv
        self.attention_scale = attention_scale
        self.softcap = softcap

        self.add_module(
            "attn_norm", RMSNormLayer(theta("attn_norm"), epsilon=rms_epsilon)
        )
        self.add_module("attn_q", LinearLayer(theta("attn_q")))
        self.add_module("attn_k", LinearLayer(theta("attn_k")))
        self.add_module("attn_v", LinearLayer(theta("attn_v")))
        self.add_module("attn_output", LinearLayer(theta("attn_output")))

        if theta.optional_tensor("attn_output_norm") is None:
            self.add_module(
                "attn_output_norm",
                torch.nn.Identity(),
            )
        else:
            self.add_module(
                "attn_output_norm",
                RMSNormLayer(theta("attn_output_norm"), epsilon=rms_epsilon),
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

        xk, xv = self.transact_cache(
            xk_cache_update=xk,
            xv_cache_update=xv,
            seq_block_ids=seq_block_ids,
            kv_seq_len=kv_seq_len,
            start_positions=start_positions,
            cache_state=cache_state,
            xk_temp=xk_temp,
            xv_temp=xv_temp,
        )

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

        attn_weights = ops.matmul(xq, keys.transpose(2, 3))
        if self.attention_scale is None:
            attn_weights = attn_weights / math.sqrt(self.head_dim)
        else:
            attn_weights = attn_weights * self.attention_scale

        # Flash attention.
        if self.softcap is not None:
            attn_weights = self.softcap * torch.tanh(attn_weights / self.softcap)

        self.assert_not_nan(attn_weights)

        # Apply attention mask.
        self.trace_tensor("attn_weights", attn_weights, values=False)
        if attention_mask is not None:
            # self.trace_tensor("attn_mask", attention_mask)
            attn_weights = attn_weights + attention_mask

        attn_weights = ops.softmax(ops.to(attn_weights, dtype=torch.float32), dim=-1)
        attn_weights = ops.to(attn_weights, dtype=xq.dtype)
        attn_output = ops.matmul(attn_weights, values)  # (bs, heads, slen, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bs, batch_seq_len, -1)

        # Project.
        attn_output = self.attn_output(attn_output)
        attn_output = self.attn_output_norm(attn_output)

        h = h + attn_output
        return h

    def transact_cache(
        self,
        *,
        xk_cache_update: torch.Tensor,
        xv_cache_update: torch.Tensor,
        cache_state: list[torch.Tensor],
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: Optional[torch.Tensor],
        kv_seq_len: int,
        start_positions: Optional[torch.Tensor] = None,
        xk_temp: Optional[torch.Tensor] = None,
        xv_temp: Optional[torch.Tensor] = None,
    ):
        cache = self.cache
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
        xk, xv = cache.read(
            cache_state,
            read_into_partitions=[
                xk_temp[:, 0:kv_seq_len, ...],
                xv_temp[:, 0:kv_seq_len, ...],
            ],
            transformer_block_index=self.block_index,
            page_ids=seq_block_ids,
            seq_len=kv_seq_len,
        )

        # For computation, we create a subview of the xk/xv tensors to have
        # a sequence length covering the blocked size. This must include
        # the newly added row (the caller is responsible for ensuring that
        # every block has at least one row left). We'll compute on this
        # ragged view and use an appropriate mask.
        return xk, xv
