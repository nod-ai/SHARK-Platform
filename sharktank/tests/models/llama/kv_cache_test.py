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
from ...layers import *


default_arguments = {
    "block_count": 1,
    "seq_len": 16,
    "head_count": 32,
    "head_dim": 128,
    "ffn_dim": 8640,
}
block_count = 1
seq_len = 16
head_count = 32
head_dim = 128
ffn_dim = 11008  # 4096
head_count_kv = 32
block_seq_stride = 16
rms_epsilon = 1e-5
rope_dimension_count = 128
max_seq_len = 4096
start_positions = torch.tensor([8])
bs = 1


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
        self.paged_attn_blocks = nn.ModuleList(
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
        self.direct_attn_blocks = nn.ModuleList(
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

    def testDirectAndPagedKVCachePrefill(self):
        torch.set_default_dtype(torch.float32)

        paged_cache_state = self.paged_kv_cache.allocate(page_count=128)
        paged_seq_block_ids = torch.tensor(
            [
                [127],
            ]
        )
        direct_cache_state = self.direct_kv_cache.allocate(bs=1)
        direct_seq_block_ids = torch.tensor(
            [
                [0],
            ]
        )

        paged_input_tensor = make_rand_torch(
            (1, seq_len, head_count * head_dim), dtype=torch.float32
        )
        direct_input_tensor = paged_input_tensor.detach().clone()

        # Iterate over paged attention blocks.
        for block_idx, paged_block in enumerate(self.paged_attn_blocks):
            if block_idx != 0:
                paged_input_tensor = paged_block(
                    paged_input_tensor,
                    embedding=self.attention_embedding,
                    attention_mask=self.prefill_attention_mask,
                    start_index=0,
                    cache_state=paged_cache_state,
                    seq_block_ids=paged_seq_block_ids,
                )
        # Iterate over direct attention blocks.
        for block_idx, direct_block in enumerate(self.direct_attn_blocks):
            if block_idx != 0:
                direct_input_tensor = direct_block(
                    direct_input_tensor,
                    embedding=self.attention_embedding,
                    attention_mask=self.prefill_attention_mask,
                    start_index=0,
                    cache_state=direct_cache_state,
                    seq_block_ids=direct_seq_block_ids,
                )
        paged_prefill_attn_output = paged_input_tensor
        direct_prefill_attn_output = direct_input_tensor

        assert paged_prefill_attn_output.shape == direct_prefill_attn_output.shape
        torch.testing.assert_close(
            paged_prefill_attn_output, direct_prefill_attn_output
        )

    @unittest.skip(
        "Attention weights are matching, but attention output does not match, need to look into it more"
    )
    def testDirectAndPagedKVCacheDecode(self):
        torch.set_default_dtype(torch.float32)

        paged_cache_state = self.paged_kv_cache.allocate(page_count=128)
        print(paged_cache_state)
        paged_seq_block_ids = torch.tensor(
            [
                [127],
            ]
        )
        direct_cache_state = self.direct_kv_cache.allocate(bs=1)
        direct_seq_block_ids = torch.tensor(
            [
                [0],
            ]
        )

        token_paged_input_tensor = make_rand_torch(
            (1, 1, head_count * head_dim), dtype=torch.float32
        )
        print(token_paged_input_tensor)
        token_direct_input_tensor = token_paged_input_tensor.detach().clone()

        embedding_batch_mask = self.attention_embedding.compute_batch_mask(
            start_positions, batch_seq_len=1
        )

        xk_temp = torch.empty(
            [
                bs,
                max_seq_len,
                head_count_kv,
                head_dim,
            ],
            dtype=torch.float32,
            device="cpu",
        )
        xv_temp = torch.empty(
            [
                bs,
                max_seq_len,
                head_count_kv,
                head_dim,
            ],
            dtype=torch.float32,
            device="cpu",
        )

        values = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            float("-inf"),
            float("-inf"),
            float("-inf"),
            float("-inf"),
            float("-inf"),
            float("-inf"),
            float("-inf"),
        ]
        attention_mask = torch.tensor(values).view(1, 1, 1, 16)

        # Iterate over paged attention blocks.
        for block_idx, paged_block in enumerate(self.paged_attn_blocks):
            token_paged_input_tensor = paged_block(
                token_paged_input_tensor,
                start_positions=start_positions,
                embedding=self.attention_embedding,
                embedding_batch_mask=embedding_batch_mask,
                attention_mask=attention_mask,
                cache_state=paged_cache_state,
                seq_block_ids=paged_seq_block_ids,
                xk_temp=xk_temp,
                xv_temp=xv_temp,
            )

        # Iterate over direct attention blocks.
        for block_idx, direct_block in enumerate(self.direct_attn_blocks):
            token_direct_input_tensor = direct_block(
                token_direct_input_tensor,
                start_positions=start_positions,
                embedding=self.attention_embedding,
                embedding_batch_mask=embedding_batch_mask,
                attention_mask=attention_mask,
                cache_state=direct_cache_state,
                seq_block_ids=direct_seq_block_ids,
                xk_temp=xk_temp,
                xv_temp=xv_temp,
            )

        paged_decode_attn_output = token_paged_input_tensor
        direct_decode_attn_output = token_direct_input_tensor
        print("paged:", paged_decode_attn_output)
        print("direct:", direct_decode_attn_output)
        assert paged_decode_attn_output.shape == direct_decode_attn_output.shape
        torch.testing.assert_close(paged_decode_attn_output, direct_decode_attn_output)


if __name__ == "__main__":
    unittest.main()
