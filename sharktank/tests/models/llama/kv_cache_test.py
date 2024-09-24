# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import torch
import torch.nn as nn
from sharktank.models.llama.llama import (
    PagedLlamaAttentionBlock,
    PagedKVCache,
    DirectKVCache,
)
from sharktank.models.llama.testing import *
from sharktank.layers.rotary_embedding import RotaryEmbeddingLayer
from sharktank.layers import causal_llm


class KVCacheTest(unittest.TestCase):
    def setUp(self):
        self.block_count = 5
        self.seq_len = 16
        self.head_count = 32
        self.head_dim = 128
        self.ffn_dim = 11008
        self.head_count_kv = 32
        self.block_seq_stride = 16
        self.rms_epsilon = 1e-5
        self.rope_dimension_count = 128
        self.max_seq_len = 4096
        self.start_positions = torch.tensor([8])
        self.bs = 1
        self.device = "cpu"
        self.attention_dtype = torch.float32
        self.attention_block_theta = make_attention_block_theta(
            feature_dim=self.head_count * self.head_dim,
            ffn_dim=self.ffn_dim,
            dtype=self.attention_dtype,
        )
        self.paged_kv_cache = PagedKVCache(
            transformer_block_count=self.head_count,
            attn_head_count=self.head_count,
            attn_head_dim=self.head_dim,
            cache_partition_count=2,  # One for each of K/V.
            block_seq_stride=self.block_seq_stride,
            device=self.device,
            dtype=self.attention_dtype,
        )
        self.direct_kv_cache = DirectKVCache(
            block_seq_stride=self.block_seq_stride,
            transformer_block_count=self.head_count,
            attn_head_count=self.head_count,
            attn_head_dim=self.head_dim,
            seq_length=self.max_seq_len,
            device=self.device,
            dtype=self.attention_dtype,
        )
        self.attention_embedding = RotaryEmbeddingLayer(
            rope_dimension_count=self.rope_dimension_count,
            max_seqlen=self.max_seq_len,
            device=self.device,
            use_hf=False,
        )
        self.paged_attn_blocks = nn.ModuleList(
            [
                PagedLlamaAttentionBlock(
                    self.attention_block_theta,
                    block_index=n,
                    cache=self.paged_kv_cache,
                    head_count=self.head_count,
                    head_dim=self.head_dim,
                    head_count_kv=self.head_count_kv,
                    rms_epsilon=self.rms_epsilon,
                )
                for n in range(self.block_count)
            ]
        )
        self.direct_attn_blocks = nn.ModuleList(
            [
                PagedLlamaAttentionBlock(
                    theta=self.attention_block_theta,
                    block_index=n,
                    cache=self.direct_kv_cache,
                    head_count=self.head_count,
                    head_dim=self.head_dim,
                    head_count_kv=self.head_count_kv,
                    rms_epsilon=self.rms_epsilon,
                )
                for n in range(self.block_count)
            ]
        )
        self.paged_cache_state = self.paged_kv_cache.allocate(page_count=128)
        self.paged_seq_block_ids = torch.tensor(
            [
                [127],
            ]
        )
        self.direct_cache_state = self.direct_kv_cache.allocate(bs=1)
        self.direct_seq_block_ids = torch.tensor(
            [
                [0],
            ]
        )
        self.embedding_batch_mask = self.attention_embedding.compute_batch_mask(
            self.start_positions, batch_seq_len=1
        )
        self.model = causal_llm.BaseCausalLMModel(
            self.attention_block_theta, context_length=self.max_seq_len
        )
        self.prefill_attention_mask = self.model.attention_mask(
            self.model.input_mask(self.start_positions, self.seq_len)
        )

    def testDirectAndPagedKVCachePrefill(self):
        torch.set_default_dtype(torch.float32)

        paged_input_tensor = make_rand_torch(
            (1, self.seq_len, self.head_count * self.head_dim),
            dtype=self.attention_dtype,
        )
        direct_input_tensor = paged_input_tensor.detach().clone()
        # Iterate over paged attention blocks.
        for block_idx, paged_block in enumerate(self.paged_attn_blocks):
            paged_input_tensor = paged_block(
                paged_input_tensor,
                embedding=self.attention_embedding,
                start_index=0,
                attention_mask=self.prefill_attention_mask,
                cache_state=self.paged_cache_state,
                seq_block_ids=self.paged_seq_block_ids,
            )
        # Iterate over direct attention blocks.
        for block_idx, direct_block in enumerate(self.direct_attn_blocks):
            direct_input_tensor = direct_block(
                direct_input_tensor,
                embedding=self.attention_embedding,
                start_index=0,
                attention_mask=self.prefill_attention_mask,
                cache_state=self.direct_cache_state,
                seq_block_ids=self.direct_seq_block_ids,
            )
        page_table = self.paged_kv_cache.unflatten_page_table(self.paged_cache_state)
        index_written = self.start_positions.item()
        """
            Getting the value of the paged_seq_block_ids, which is the page id we are writing
            the K/V cache into.
        """
        page_id = self.paged_seq_block_ids[0][0].item()
        """
            direct_cache_state is a list of num_transformer_blocks * 2 (one for K and one for V),
            so here we index into the first transformer block's keys with self.direct_cache_state[0]
            and the first transformer block's values with self.direct_cache_state[1]. Each row
            in direct_cache_state is a tensor of [bs, seq_len , attn_heads, attn_dim], so we make sure
            the first 8 (start_position) tensors starting at sequence 0 of the seq_len are written to.
        """
        updated_direct_cache_state = self.direct_cache_state[0][
            :, :index_written
        ].squeeze(0)
        """
            paged_cache_state is a list of a single tensor that represents a flattened page table.
            Indexing into self.paged_cache_state[0] and unflattening the page table columns to a 6D tensor of:
                * transformer block
                * cache partition (K or V cache)
                * block sequence stride (number of sequence positions per block)
                * attention heads
                * attention dimensionality
            allows us to access the cache partitions for a certain transformer block and sequence in a
            certain page_id. For example, page_table[page_id][0, 0, :index_written] lets us access the
            first transformer block's K cache for the first 8 (start_positions) tensors starting at
            sequence 0.
        """
        updated_paged_cache_state = page_table[page_id][0, 0, :index_written]
        assert updated_direct_cache_state.shape == updated_paged_cache_state.shape
        torch.testing.assert_close(
            updated_direct_cache_state, updated_paged_cache_state
        )

        paged_prefill_attn_output = paged_input_tensor
        direct_prefill_attn_output = direct_input_tensor
        assert paged_prefill_attn_output.shape == direct_prefill_attn_output.shape
        torch.testing.assert_close(
            paged_prefill_attn_output, direct_prefill_attn_output
        )

    @unittest.skip(
        "Bug in Windows decode test for paged_decode_attn_output vs. direct_decode_attn_output"
    )
    def testDirectAndPagedKVCacheDecode(self):
        torch.set_default_dtype(torch.float32)
        self.start_positions.add_(1)
        assert self.direct_seq_block_ids.shape[1] == self.paged_seq_block_ids.shape[1]
        decode_attention_mask = self.model.decode_attention_mask(
            self.model.input_mask(
                self.start_positions, self.direct_seq_block_ids.shape[1] * self.seq_len
            )
        )

        token_paged_input_tensor = make_rand_torch(
            (1, 1, self.head_count * self.head_dim), dtype=self.attention_dtype
        )
        token_direct_input_tensor = token_paged_input_tensor.detach().clone()

        xk_temp = torch.empty(
            [
                self.bs,
                self.max_seq_len,
                self.head_count_kv,
                self.head_dim,
            ],
            dtype=self.attention_dtype,
            device=self.device,
        )
        xv_temp = torch.empty(
            [
                self.bs,
                self.max_seq_len,
                self.head_count_kv,
                self.head_dim,
            ],
            dtype=self.attention_dtype,
            device=self.device,
        )

        # Iterate over paged attention blocks.
        for block_idx, paged_block in enumerate(self.paged_attn_blocks):
            token_paged_input_tensor = paged_block(
                token_paged_input_tensor,
                start_positions=self.start_positions,
                embedding=self.attention_embedding,
                embedding_batch_mask=self.embedding_batch_mask,
                attention_mask=decode_attention_mask,
                cache_state=self.paged_cache_state,
                seq_block_ids=self.paged_seq_block_ids,
                xk_temp=xk_temp,
                xv_temp=xv_temp,
            )

        # Iterate over direct attention blocks.
        for block_idx, direct_block in enumerate(self.direct_attn_blocks):
            token_direct_input_tensor = direct_block(
                token_direct_input_tensor,
                start_positions=self.start_positions,
                embedding=self.attention_embedding,
                embedding_batch_mask=self.embedding_batch_mask,
                attention_mask=decode_attention_mask,
                cache_state=self.direct_cache_state,
                seq_block_ids=self.direct_seq_block_ids,
                xk_temp=xk_temp,
                xv_temp=xv_temp,
            )

        page_table = self.paged_kv_cache.unflatten_page_table(self.paged_cache_state)
        index_written = self.start_positions.item()
        page_id = self.paged_seq_block_ids[0][0].item()
        updated_direct_cache_state_keys = self.direct_cache_state[0][
            :, index_written
        ].squeeze(0)
        updated_paged_cache_state_keys = page_table[page_id][0, 0, index_written]
        updated_direct_cache_state_values = self.direct_cache_state[1][
            :, index_written
        ].squeeze(0)
        updated_paged_cache_state_values = page_table[page_id][0, 1, index_written]
        assert (
            updated_direct_cache_state_keys.shape
            == updated_paged_cache_state_keys.shape
        )
        torch.testing.assert_close(
            updated_direct_cache_state_keys, updated_paged_cache_state_keys
        )
        assert (
            updated_direct_cache_state_values.shape
            == updated_paged_cache_state_values.shape
        )
        torch.testing.assert_close(
            updated_direct_cache_state_values, updated_paged_cache_state_values
        )

        paged_decode_attn_output = token_paged_input_tensor
        direct_decode_attn_output = token_direct_input_tensor
        assert paged_decode_attn_output.shape == direct_decode_attn_output.shape
        torch.testing.assert_close(paged_decode_attn_output, direct_decode_attn_output)


if __name__ == "__main__":
    unittest.main()
