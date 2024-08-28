# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

from dataclasses import dataclass
import math
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from shark_turbine.aot import *

from ..layers import *
from ..types import Theta

from ..utils import cli

import sys

################################################################################
# Config
################################################################################

@dataclass
class RefLlamaModelConfig:
    context_length=4096
    embedding_length=4096
    block_count=32
    feed_forward_length=11008
    rope_dimension_count=128
    attention_head_count=32
    attn_head_dim=128
    attention_layer_norm_rms_epsilon=9.999999747378752e-06
    attention_head_count_kv=32

    # local params
    sl = 1
    start_index = 0
    q_len = 1
    feature_dim = 4096
    kv_seq_len = 1 
    head_dim = 128

    # Dtype to use for general FP activations not otherwise configured.
    activation_dtype: torch.dtype = torch.float16

class LlamaAttentionBlock(torch.nn.Module):
    """Implements a self attention layer in the style of Llama."""

    def __init__(
        self,
        config: RefLlamaModelConfig,
    ):
        super().__init__()

        self.config = config
        self.activation_dtype: torch.dtype = torch.float16

    def create_cache(self, bs: int) -> list[torch.Tensor]:
        return [
            torch.empty(
                (
                    bs,
                    self.config.context_length,
                    self.config.attention_head_count,
                    self.config.rope_dimension_count,
                ),
                dtype=self.config.activation_dtype,
            )
            for _ in range(self.config.block_count * 2)
        ]

    def sdpa(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        is_causal: bool,
        start_index: Optional[int] = 0,
        attention_mask: Optional[torch.Tensor] = None,
    ):

        bs, q_len, _, _ = xq.shape
        kv_seq_len = start_index + q_len

        # TODO: Some model variants do some form of kv repetition to expand the
        # count of kv heads to the count of attention heads used by the q.
        # assert self.head_count == self.head_count_kv, "NYI: KV expansion"

        # Update our positions in the cache.

        cache_k[:bs, start_index:kv_seq_len] = xk
        cache_v[:bs, start_index:kv_seq_len] = xv

        # Derive keys/values from the entirety of the available sequence.
        keys = cache_k[:bs, :kv_seq_len]
        values = cache_v[:bs, :kv_seq_len]

        # Tranpose into [bs, heads, sl, dim]
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(xq, keys, values, attn_mask=attention_mask, is_causal=is_causal)
        attn_output = attn_output.transpose(1, 2).reshape(bs, q_len, -1)
        return attn_output

def main(args: list[str]):

    torch.no_grad().__enter__()

    parser = cli.create_parser()
    # cli.add_input_dataset_options(parser)
    parser.add_argument(
        "--output-mlir",
        help="Output file path for exported MLIR file",
        default="/tmp/batch_llama_v1.mlir",
    )
    parser.add_argument(
        "--output-config",
        help="Output file path for exported config file",
        default="/tmp/batch_llama_v1.json",
    )
    parser.add_argument(
        "--bs",
        help="Comma-separated batch size(s) to generate, e.g. `4` or `2,4`",
        type=lambda arg: [int(bs) for bs in arg.split(",")],
        default="2,4",
    )
    parser.add_argument(
        "--verbose",
        help="Include verbose logging",
        action="store_true",
    )
    args = cli.parse(parser)

    # dataset = cli.get_input_dataset(args)
    # hp = configs.LlamaHParams.from_gguf_props(dataset.properties)

    def generate_params_json(hp, sdpa_bs: list[int]):
        return {
            "module_name": "module",
            "module_abi_version": 1,
            "max_seq_len": hp.context_length,
            "attn_head_count": hp.attention_head_count,
            "attn_head_dim": hp.attn_head_dim,
            "batch_sizes": hp.bs,
            "transformer_block_count": hp.block_count,
        }

    hp = RefLlamaModelConfig()
    hp.activation_dtype = torch.float16
    start_index = 0
    hp.bs = args.bs
    sdpa_block = LlamaAttentionBlock(hp)

    bsizes = []
    for bs in hp.bs:
        q = torch.zeros((bs, 1, 32, 128), dtype=torch.float16)
        k = torch.zeros((bs, 1, 32, 128), dtype=torch.float16)
        v = torch.zeros((bs, 1, 32, 128), dtype=torch.float16)

        block_cache_k = torch.zeros((bs, 4096, 32, 128), dtype=torch.float16)
        block_cache_v = torch.zeros((bs, 4096, 32, 128), dtype=torch.float16)

        dtype = q.dtype
        attention_mask = None
        if hp.sl > 1:
            # Use the smallest value like HF as opposed to -inf like original.
            # A little bit easier for some systems.
            attention_mask = torch.full(
                (1, 1, hp.sl, hp.sl), torch.finfo(dtype).min, dtype=dtype
            )
            attention_mask = torch.triu(
                attention_mask, diagonal=start_index + 1
            ).type_as(q)

        print(f"Exporting sdpa{bs}")
        fxb = FxProgramsBuilder(sdpa_block)
        example_args = (q, k, v, block_cache_k, block_cache_v)

        # dynamic_shapes = {
        #         "seq_lens": {},
        #         "seq_block_ids": {1: block_dim},
        #         "cache_state": cache_state_dynamic_shapes,
        #     }

        # @fxb.export_program(
        #         name=f"sdpa{hp.bs}",
        #         args=(hp, is_causal, attention_mask),
        # )
        @fxb.export_program(
            name=f"sdpa{bs}",
            args=example_args,
        )
        def _(sdpa_block, q, k, v, block_cache_k, block_cache_v):
            attention_mask = None
            h = sdpa_block.sdpa(
                xq=q,
                xk=k,
                xv=v,
                cache_k=block_cache_k,
                cache_v=block_cache_v,
                is_causal = False,
                attention_mask=attention_mask,
            )
            return h
        
        bsizes.append(bs)

    config = generate_params_json(hp, bsizes)
    print("GENERATED!")

    # if args.verbose:
    #     for name, ep in fxb.programs.items():
    #         print(f"EXPORT {name}:\n{ep}")

    print("Exporting")
    output = export(fxb)
    print(f"Saving to '{args.output_mlir}'")
    output.save_mlir(args.output_mlir)
    json.dump(config, open(args.output_config, "w"))
    

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
