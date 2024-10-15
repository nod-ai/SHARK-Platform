# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from typing import List

import torch

from sharktank.models.llama.testing import *
from sharktank.layers.rotary_embedding import RotaryEmbeddingLayer
from sharktank.models.llama.llama import AttentionFFNBlock, PagedKVCache
from sharktank import ops

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaDecoderLayer,
)
from transformers.models.llama.configuration_llama import LlamaConfig

import unittest
from parameterized import parameterized

import torch

from iree.turbine import aot
from sharktank import kernels
from sharktank.types import layout_utils


class DynamicSDPATest(unittest.TestCase):
    def test(self):
        class MyModule(torch.nn.Module):
            def forward(self, query, key, value, attention_mask):
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query=query,  # [bs, ..., sl, dim]
                    key=key,  # [bs, ..., sl, dim]
                    value=value,  # [bs, ..., sl, dim]
                    attn_mask=attention_mask,  # [bs, ..., sl, sl]
                    dropout_p=0.0,
                    is_causal=False,  # assumes causal masking when true
                    scale=None,  # defaults to 1/sqrt(dim)
                )
                return attn_output

        mod = MyModule()
        batch = torch.export.Dim("batch")
        words = torch.export.Dim("words")
        ep = torch.export.export(
            mod,
            args=(
                torch.rand([16, 16, 16], dtype=torch.float32),
                torch.rand([16, 16, 16], dtype=torch.float32),
                torch.rand([16, 16, 16], dtype=torch.float32),
                torch.rand([16, 16, 16], dtype=torch.float32),
            ),
            dynamic_shapes={
                "query": {0: batch, 1: words},
                "key": {0: batch, 1: words},
                "value": {0: batch, 1: words},
                "attention_mask": {0: batch, 1:words},
            },
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        print(asm)
        self.assertIn("@sharktank_einsum_2args_q4_mek_menk_men_32_f32", asm)


if __name__ == "__main__":
    unittest.main()
