# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..layers import *


def create_kv_cache(config: LlamaModelConfig) -> BaseKVCache:
    hp = config.hp
    if config.kv_cache_type == "direct":
        return DirectKVCache(
            block_seq_stride=config.block_seq_stride,
            transformer_block_count=hp.block_count,
            attn_head_count=hp.attention_head_count_kv,
            attn_head_dim=hp.attn_head_dim,
            seq_length=hp.context_length,
            device=config.device,
            dtype=config.attention_dtype,
        )
    elif config.kv_cache_type == "paged":
        return PagedKVCache(
            transformer_block_count=hp.block_count,
            attn_head_count=hp.attention_head_count_kv,
            attn_head_dim=hp.attn_head_dim,
            cache_partition_count=2,  # One for each of K/V.
            block_seq_stride=config.block_seq_stride,
            device=config.device,
            dtype=config.attention_dtype,
            shard_count=config.tensor_parallelism_size,
        )
    else:
        raise NotImplementedError(f"kv_cache_type = {config.kv_cache_type}")
