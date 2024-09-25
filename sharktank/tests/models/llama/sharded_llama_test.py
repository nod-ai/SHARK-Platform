# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from sharktank.models.llama.llama import LlamaModelConfig, PagedLlamaModelV1
import sharktank.ops as ops
from sharktank.models.llama.testing import make_random_llama_theta
from sharktank.models.llama.sharding import shard_theta
from sharktank.layers.configs import LlamaHParams
from sharktank.utils.math import round_up_to_multiple_of
import torch
from copy import deepcopy


class AttentionBlockTest(unittest.TestCase):
    def testToyModelCompareToUnsharded(self):
        """Run a sharded variant of a toy model size and compare it against the
        unsharded variant."""
        torch.random.manual_seed(123456)
        dtype = torch.float32
        torch.set_default_dtype(dtype)
        batch_size = 3
        attention_head_count_kv = 4
        attention_head_count = attention_head_count_kv * 5
        vocabulary_size = 19
        rope_dimension_count = 7 * 2
        attn_head_dim = rope_dimension_count
        block_seq_stride = 13
        cache_page_count = 11
        config = LlamaModelConfig(
            hp=LlamaHParams(
                context_length=block_seq_stride * 2,
                embedding_length=attention_head_count * attn_head_dim,
                block_count=3,
                feed_forward_length=23,
                rope_dimension_count=rope_dimension_count,
                rope_freq_base=500000.0,
                attention_head_count=attention_head_count,
                attn_head_dim=attn_head_dim,
                attention_layer_norm_rms_epsilon=0.01,
                attention_head_count_kv=attention_head_count_kv,
                expert_count=0,
                expert_used_count=0,
            ),
            block_seq_stride=block_seq_stride,
            activation_dtype=dtype,
            attention_dtype=dtype,
        )
        theta = make_random_llama_theta(
            config=config,
            vocab_size=vocabulary_size,
        )

        model = PagedLlamaModelV1(theta, config)
        seq_lens = torch.randint(high=config.hp.context_length + 1, size=[batch_size])
        seq_lens[batch_size - 1] = config.hp.context_length
        cache_state = model.cache.paged.allocate(page_count=cache_page_count)
        cache_state = [torch.rand_like(cache_state[0])]
        cache_state_snapshot = deepcopy(cache_state)
        batch_seq_len = round_up_to_multiple_of(
            int(torch.max(seq_lens)), model.cache.pad_sequence_stride
        )
        token_ids = torch.randint(
            low=0,
            high=vocabulary_size,
            size=[batch_size, batch_seq_len],
            dtype=torch.int32,
        )
        attention_mask = model.attention_mask(model.input_mask(seq_lens, batch_seq_len))
        seq_block_ids = torch.arange(
            batch_size * batch_seq_len // config.block_seq_stride
        ).view(batch_size, -1)

        # Verify prefill step.
        sharded_config = deepcopy(config)
        sharded_config.tensor_parallelism_size = 2
        sharded_theta = shard_theta(theta, sharded_config)
        sharded_model = PagedLlamaModelV1(sharded_theta, sharded_config)
        sharded_cache_state = sharded_model.cache.paged.allocate(
            page_count=cache_page_count
        )
        sharded_cache_state = sharded_model.cache.paged.shard_state(
            deepcopy(cache_state)
        )

        expected_prefill_result = model.prefill(
            token_ids,
            attention_mask=attention_mask,
            seq_block_ids=seq_block_ids,
            cache_state=cache_state,
        )
        sharded_prefill_result = sharded_model.prefill(
            token_ids,
            attention_mask=attention_mask,
            seq_block_ids=seq_block_ids,
            cache_state=sharded_cache_state,
        )
        # The errors are quite high, but for float64 both errors drop to < 1e-12.
        # The numerics are probably correct.
        torch.testing.assert_close(
            sharded_prefill_result, expected_prefill_result, atol=1e-3, rtol=1e-2
        )
        expected_cache_state = cache_state[0]
        actual_cache_state = ops.unshard(
            sharded_model.cache.paged.unflatten_page_table(sharded_cache_state)
        ).flatten(start_dim=1)
        torch.testing.assert_close(
            actual_cache_state, expected_cache_state, atol=1e-4, rtol=1e-1
        )

        # Verify decode step.
        decode_token_ids = torch.randint(
            low=0,
            high=vocabulary_size,
            size=[batch_size, 1],
            dtype=torch.int32,
        )
        decode_seq_lens = torch.randint(
            high=config.hp.context_length - 2, size=[batch_size]
        )
        start_positions = decode_seq_lens + 1
        decode_batch_seq_len = round_up_to_multiple_of(
            int(torch.max(seq_lens)), model.cache.pad_sequence_stride
        )
        decode_attention_mask = model.decode_attention_mask(
            model.input_mask(decode_seq_lens, decode_batch_seq_len)
        )
        decode_cache_state = deepcopy(cache_state_snapshot)
        decode_sharded_cache_state = sharded_model.cache.paged.shard_state(
            deepcopy(decode_cache_state)
        )
        expected_decode_result = model.decode(
            decode_token_ids,
            attention_mask=decode_attention_mask,
            start_positions=start_positions,
            seq_block_ids=seq_block_ids,
            cache_state=decode_cache_state,
        )
        sharded_decode_result = sharded_model.decode(
            decode_token_ids,
            attention_mask=decode_attention_mask,
            start_positions=start_positions,
            seq_block_ids=seq_block_ids,
            cache_state=decode_sharded_cache_state,
        )
        torch.testing.assert_close(sharded_decode_result, expected_decode_result)
        expected_decode_cache_state = decode_cache_state[0]
        actual_decode_cache_state = ops.unshard(
            sharded_model.cache.paged.unflatten_page_table(decode_sharded_cache_state)
        ).flatten(start_dim=1)
        torch.testing.assert_close(
            actual_decode_cache_state, expected_decode_cache_state, atol=1e-4, rtol=1e-4
        )
