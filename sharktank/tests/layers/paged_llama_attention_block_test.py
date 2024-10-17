# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest

import torch

from iree.turbine import aot
from sharktank.layers import PagedLlamaAttentionBlock, PagedKVCache, RotaryEmbeddingLayer
from sharktank.layers.testing import make_llama_attention_block_theta


class PagedLlamaAttentionBlockTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12345)
        self.transformer_block_count = 13
        self.block_index = 1
        self.shard_count = 3
        self.head_count_kv = 2 * self.shard_count
        self.attention_head_count = 5 * self.head_count_kv
        self.attention_head_dim = 11 * 2
        self.rms_epsilon = 0.01
        self.block_seq_stride = 17
        self.cache_partition_count = 2
        self.page_count = 23
        self.embedding_length = self.attention_head_count * self.attention_head_dim
        self.rope_dimension_count = self.attention_head_dim
        self.block_seqlen = 7
        self.max_seqlen = self.block_seq_stride * self.block_seqlen
        self.rope_freq_base = None
        self.batch_size = 3
        self.start_index = 0


    def testExportDecomposed(self):
        dtype = torch.float32
        torch.set_default_dtype(dtype)

        cache = PagedKVCache(
            transformer_block_count=self.transformer_block_count,
            attn_head_count=self.head_count_kv,
            attn_head_dim=self.attention_head_dim,
            cache_partition_count=self.cache_partition_count,
            block_seq_stride=self.block_seq_stride,
            dtype=dtype,
        )

        theta = make_llama_attention_block_theta(
            head_count=self.attention_head_count,
            head_count_kv=self.head_count_kv,
            head_dim=self.attention_head_dim,
            embedding_length=self.embedding_length,
        )
        attn = PagedLlamaAttentionBlock(
            theta=theta,
            block_index=self.block_index,
            cache=cache,
            head_count=self.attention_head_count,
            head_dim=self.attention_head_dim,
            head_count_kv=self.head_count_kv,
            rms_epsilon=self.rms_epsilon,
            attention_kernel="decomposed",
        )
        seq_block_ids = torch.arange(self.batch_size * self.block_seqlen).view(
            self.batch_size, -1
        )
        embedding_module = RotaryEmbeddingLayer(
            rope_dimension_count=self.rope_dimension_count,
            max_seqlen=self.max_seqlen,
            rope_freq_base=self.rope_freq_base,
        )
        class MyModule(torch.nn.Module):
            def forward(self, h):
                return attn.forward(h, seq_block_ids=seq_block_ids, embedding=embedding_module)
        mod = MyModule()
        #fxb = aot.FxProgramsBuilder(mod)
        #@fxb.export_program(
        #        name="forward",
        #        args=(torch.rand([4,16,320]),),
        #        dynamic_shapes={})
        #def _(model, *args, **kwargs) -> torch.Tensor:
        #    return model.forward(*args, **kwargs)
        ep = torch.export.export(
            mod,
            args=(
                torch.rand([4, 16, 320], dtype=torch.float32),
            ),
        )
        output = aot.export(fxb)
        #output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertNotIn("scaled_dot_product_attention", asm)


if __name__ == "__main__":
    unittest.main()
