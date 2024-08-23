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
    bs = 1
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
      #   super().__init__(theta)
        # self.add_module(
        #     "attn_norm", RMSNormLayer(theta("attn_norm"), epsilon=rms_epsilon)
        # )
        # self.add_module("attn_q", LinearLayer(theta("attn_q")))
        # self.add_module("attn_k", LinearLayer(theta("attn_k")))
        # self.add_module("attn_v", LinearLayer(theta("attn_v")))
        # self.add_module("attn_output", LinearLayer(theta("attn_output")))
        # self.add_module(
        #     "ffn_norm", RMSNormLayer(theta("ffn_norm"), epsilon=rms_epsilon)
        # )
        # self.add_module("ffn_gate", LinearLayer(theta("ffn_gate")))
        # self.add_module("ffn_up", LinearLayer(theta("ffn_up")))
        # self.add_module("ffn_down", LinearLayer(theta("ffn_down")))

        # self.embedding = embedding
        # self.head_count = head_count
        # self.head_dim = head_dim
        # self.head_count_kv = head_count_kv
        # self.context_length = context_length
        # self.attention_head_count = attention_head_count
        # self.rope_dimension_count = head_dim

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
        # x = self.attn_norm(h)

        # bs, q_len, feature_dim = x.shape
        kv_seq_len = self.config.kv_seq_len
        bs = self.config.bs
        q_len = self.config.q_len
        # assert feature_dim == self.head_count * self.head_dim

        # xq = self.attn_q(x)
        # xk = self.attn_k(x)
        # xv = self.attn_v(x)

        # xq = xq.view(bs, q_len, self.head_count, self.head_dim)
        # xk = xk.view(bs, q_len, self.head_count_kv, self.head_dim)
        # xv = xv.view(bs, q_len, self.head_count_kv, self.head_dim)

        # xq, xk = self.embedding(xq=xq, xk=xk, start_index=start_index)

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

        # # Flash attention.
        # attn_weights = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # # Apply attention mask.
        # if attention_mask is not None:
        #     expected_mask_shape = (bs, 1, q_len, kv_seq_len)
        #     assert (
        #         attention_mask.shape == expected_mask_shape
        #     ), f"Attention mask should be of size {expected_mask_shape}, but is {attention_mask.shape}"
        #     attn_weights = attn_weights + attention_mask

        # attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(xq)
        # attn_output = torch.matmul(attn_weights, values)  # (bs, heads, slen, head_dim)
        attn_output = F.scaled_dot_product_attention(xq, keys, values, attn_mask=attention_mask, is_causal=is_causal)
        print('\n\n\nattn_output', attn_output.shape)
        attn_output = attn_output.transpose(1, 2).reshape(bs, q_len, -1)

        # Project.
        # attn_output = self.attn_output(attn_output)
      
      #   print(attn_output)
      #   print(xq)
      #   print(attn_weights)

        # # Remainder of the block.
        # h = h + attn_output

        # # Feed forward network.
        # ffn_input = self.ffn_norm(h)
        # ffn_gate = F.silu(self.ffn_gate(ffn_input))
        # ffn_up = self.ffn_up(ffn_input)
        # ffn_down = self.ffn_down(ffn_gate * ffn_up)
        # return h + ffn_down
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
        default="4",
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

    is_causal = True
    q = torch.zeros((1, 1, 32, 128), dtype=torch.float16)
    k = torch.zeros((1, 1, 32, 128), dtype=torch.float16)
    v = torch.zeros((1, 1, 32, 128), dtype=torch.float16)

    block_cache_k = torch.zeros((1, 4096, 32, 128), dtype=torch.float16)
    block_cache_v = torch.zeros((1, 4096, 32, 128), dtype=torch.float16)
    
    # model = DirectCacheLlamaModelV1(dataset.root_theta, ref_llama_config)

    sdpa_block = LlamaAttentionBlock(hp)

    # next_tokens = [1, 1059, 31871, 1217, 322, 266, 3682, 6075, 31902, 13, 31849, 31871]
    # print(f"Step {start_index}")

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

    # h = sdpa_block.sdpa(
    #     xq=q,
    #     xk=k,
    #     xv=v,
    #     cache_k=block_cache_k,
    #     cache_v=block_cache_v,
    #     start_index=0,
    #     is_causal=is_causal,
    #     attention_mask=attention_mask,
    # )

    # print('\n\nFinal output: h\n', h)

    # print(f"  : tokens = {tokens}")

    # Decode a step.
    # print("Decoding...")
    # print(tokens.shape, tokens)
    # decode_token = model.forward(tokens, start_index=12, local_kv_cache=kv_cache)
    # print(f"  : decode tokens = {decode_token}")

    print(f"Exporting sdpa{hp.bs}")
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
        name=f"sdpa{hp.bs}",
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
            is_causal = True,
            attention_mask=attention_mask,
        )
        print('\n\nFinal output: h\n', h)
        return h
    
    bsizes = hp.bs

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
