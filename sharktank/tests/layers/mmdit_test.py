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
from sharktank.layers import (
    MMDITDoubleBlock,
    PagedLlamaAttentionBlock,
    PagedKVCache,
    RotaryEmbeddingLayer,
)
import sharktank.ops as ops
from sharktank.layers.testing import (
    make_llama_attention_block_theta,
    make_mmdit_block_theta,
)
from sharktank.types.tensors import DefaultPrimitiveTensor


class MMDITTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12345)
        self.hidden_size = 3072
        self.num_heads = 24

        self.transformer_block_count = 13
        self.block_index = 1
        self.shard_count = 3
        self.head_count_kv = 2 * self.shard_count
        self.attention_head_count = 5 * self.head_count_kv
        self.attention_head_dim = 24
        self.rms_epsilon = 0.01
        self.cache_partition_count = 2
        self.page_count = 23
        self.embedding_length = self.attention_head_count * self.attention_head_dim
        self.rope_dimension_count = self.attention_head_dim
        self.block_seqlen = 7
        self.block_seq_stride = 17
        self.max_seqlen = self.block_seq_stride * self.block_seqlen
        self.rope_freq_base = None
        self.batch_size = 3
        self.start_index = 0

    def testExport(self):
        dtype = torch.float32

        txt_ids = torch.rand([self.batch_size, 3, self.max_seqlen, 3])
        img_ids = torch.rand([self.batch_size, 3, self.max_seqlen, 3])
        pe_dim = self.hidden_size // self.num_heads
        axes_dim = [16, 56, 56]
        theta = 10000

        theta = make_mmdit_block_theta()
        mmdit = MMDITDoubleBlock(
            theta=theta,
            num_heads=self.num_heads,
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
            def forward(self, img, txt, vec, pe):
                return mmdit.forward(
                    img,
                    txt,
                    vec,
                    pe,
                )

        mod = MyModule()
        img = torch.rand([self.batch_size, 1024, self.hidden_size])
        txt = torch.rand([self.batch_size, 512, self.hidden_size])
        vec = torch.rand([self.batch_size, self.hidden_size])
        rot = torch.rand([self.batch_size, 1, 1536, 64, 2, 2])
        mod.forward(img, txt, vec, rot)
        fxb = aot.FxProgramsBuilder(mod)

        @fxb.export_program(name="mmdit", args=(img, txt, vec, rot), strict=False)
        def _(model, img, txt, vec, rot) -> torch.Tensor:
            return model(img, txt, vec, rot)

        output = aot.export(fxb)
        output.verify()
        asm = str(output.mlir_module)


if __name__ == "__main__":
    unittest.main()
