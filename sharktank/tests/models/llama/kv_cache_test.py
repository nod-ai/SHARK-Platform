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

block_count = 5
seq_len = 16
head_count = 32
head_dim = 128
ffn_dim = 8640
head_count_kv = 32
block_seq_stride = 16
rms_epsilon = 0.01
rope_dimension_count = 128
max_seq_len = 4096


class KVCacheTest(unittest.TestCase):
    def setUp(self):
        self.attention_block_theta = make_attention_block_theta(
            feature_dim=head_count * head_dim, ffn_dim=ffn_dim, dtype=torch.float32
        )
        self.paged_kv_cache = PagedKVCache(
            transformer_block_count=head_count,
            attn_head_count=head_count,
            attn_head_dim=head_dim,
            cache_partition_count=2,  # One for each of K/V.
            block_seq_stride=block_seq_stride,
            device="cpu",
            dtype=torch.float32,
        )
        self.direct_kv_cache = DirectKVCache(
            block_seq_stride=block_seq_stride,
            transformer_block_count=head_count,
            attn_head_count=head_count,
            attn_head_dim=head_dim,
            seq_length=seq_len,
            device="cpu",
            dtype=torch.float32,
        )
        self.attention_embedding = RotaryEmbeddingLayer(
            rope_dimension_count=rope_dimension_count,
            max_seqlen=max_seq_len,
            device="cpu",
            use_hf=False,
        )

    def testDirectAndPagedKVCachePrefill(self):
        torch.set_default_dtype(torch.float32)

        paged_cache_state = self.paged_kv_cache.allocate(page_count=128)
        seq_block_ids = torch.tensor(
            [
                [127],
            ]
        )
        direct_cache_state = self.direct_kv_cache.allocate(bs=1)
        paged_attn_blocks = nn.ModuleList(
            [
                PagedLlamaAttentionBlock(
                    self.attention_block_theta,
                    block_index=n,
                    cache=self.paged_kv_cache,
                    head_count=head_count,
                    head_dim=head_dim,
                    head_count_kv=head_count_kv,
                    rms_epsilon=rms_epsilon,
                    use_hf=False,
                )
                for n in range(block_count)
            ]
        )
        direct_attn_blocks = nn.ModuleList(
            [
                PagedLlamaAttentionBlock(
                    theta=self.attention_block_theta,
                    block_index=n,
                    cache=self.direct_kv_cache,
                    head_count=head_count,
                    head_dim=head_dim,
                    head_count_kv=head_count_kv,
                    rms_epsilon=rms_epsilon,
                    use_hf=False,
                )
                for n in range(block_count)
            ]
        )

        paged_input_tensor = make_rand_torch(
            (1, seq_len, head_count * head_dim), dtype=torch.float32
        )
        direct_input_tensor = paged_input_tensor.detach().clone()

        # Iterate over paged attention blocks.
        for block_idx, paged_block in enumerate(paged_attn_blocks):
            if block_idx != 0:
                paged_input_tensor = paged_block(
                    paged_input_tensor,
                    embedding=self.attention_embedding,
                    start_index=0,
                    cache_state=paged_cache_state,
                    seq_block_ids=seq_block_ids,
                )
        # Iterate over direct attention blocks.
        for block_idx, direct_block in enumerate(direct_attn_blocks):
            if block_idx != 0:
                direct_input_tensor = direct_block(
                    direct_input_tensor,
                    embedding=self.attention_embedding,
                    start_index=0,
                    cache_state=direct_cache_state,
                    seq_block_ids=seq_block_ids,
                )
        paged_prefill_attn_output = paged_input_tensor
        direct_prefill_attn_output = direct_input_tensor

        assert paged_prefill_attn_output.shape == direct_prefill_attn_output.shape
        torch.testing.assert_close(
            paged_prefill_attn_output, direct_prefill_attn_output
        )

    # def testDirectAndPagedKVCacheDecode(self):


if __name__ == "__main__":
    unittest.main()
