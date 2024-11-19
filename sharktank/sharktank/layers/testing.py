# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from ..types.theta import Theta
from ..types.tensors import DefaultPrimitiveTensor
from ..utils.testing import make_rand_torch


def make_llama_attention_block_theta(
    *,
    head_count: int,
    head_count_kv: int,
    head_dim: int,
    embedding_length: int,
    dtype: torch.dtype | None = None,
) -> Theta:
    return Theta(
        {
            "attn_q.weight": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (head_count * head_dim, embedding_length), dtype=dtype
                )
            ),
            "attn_k.weight": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (head_count_kv * head_dim, embedding_length), dtype=dtype
                )
            ),
            "attn_v.weight": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (head_count_kv * head_dim, embedding_length), dtype=dtype
                )
            ),
            "attn_output.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((embedding_length, embedding_length), dtype=dtype)
            ),
            "attn_norm.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((embedding_length), dtype=dtype)
            ),
        }
    )

def make_mmdit_block_theta(
    dtype: torch.dtype | None = None
) -> Theta:
    return Theta(
        {
            "img_attn.norm.key_norm.weight": DefaultPrimitiveTensor( #
                data=make_rand_torch(
                    (128,), dtype=dtype
                )
            ),
            "img_attn.norm.query_norm.weight": DefaultPrimitiveTensor( #
                data=make_rand_torch(
                    (128,), dtype=dtype
                )
            ),
            "img_attn.proj.bias": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (3072,), dtype=dtype
                )
            ),
            "img_attn.proj.weight": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (3072, 3072), dtype=dtype
                )
            ),
            "img_attn.qkv.bias": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (9216,), dtype=dtype
                )
            ),
            "img_attn.qkv.weight": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (9216, 3072), dtype=dtype
                )
            ),
            "img_mlp.0.bias": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (12288), dtype=dtype
                )
            ),
            "img_mlp.0.weight": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (12288, 3072), dtype=dtype
                )
            ),
            "img_mlp.2.bias": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (3072), dtype=dtype
                )
            ),
            "img_mlp.2.weight": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (3072, 12288), dtype=dtype
                )
            ),
            "img_mod.lin.bias": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (18432,), dtype=dtype
                )
            ),
            "img_mod.lin.weight": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (18432, 3072), dtype=dtype
                )
            ),
            "txt_attn.norm.key_norm.weight": DefaultPrimitiveTensor( #
                data=make_rand_torch(
                    (128,), dtype=dtype
                )
            ),
            "txt_attn.norm.query_norm.weight": DefaultPrimitiveTensor( #
                data=make_rand_torch(
                    (128,), dtype=dtype
                )
            ),
            "txt_attn.proj.bias": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (3072,), dtype=dtype
                )
            ),
            "txt_attn.proj.weight": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (3072, 3072), dtype=dtype
                )
            ),
            "txt_attn.qkv.bias": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (9216,), dtype=dtype
                )
            ),
            "txt_attn.qkv.weight": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (9216, 3072), dtype=dtype
                )
            ),
            "txt_mlp.0.bias": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (12288), dtype=dtype
                )
            ),
            "txt_mlp.0.weight": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (12288, 3072), dtype=dtype
                )
            ),
            "txt_mlp.2.bias": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (3072), dtype=dtype
                )
            ),
            "txt_mlp.2.weight": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (3072, 12288), dtype=dtype
                )
            ),
            "txt_mod.lin.bias": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (18432,), dtype=dtype
                )
            ),
            "txt_mod.lin.weight": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (18432, 3072), dtype=dtype
                )
            ),
        }
    )
