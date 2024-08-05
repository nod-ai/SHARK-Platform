# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List

import torch

from ...types.tensors import *
from ...types.theta import Theta

def make_rand_torch(shape, dtype):
    return torch.rand(shape, dtype=dtype) * 2 - 1

def make_attention_block_theta(
    feature_dim: int,
    ffn_dim: int,
    dtype: torch.dtype | None = None,
) -> Theta:
    return Theta(
        {
            "self_attn.q_proj.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, feature_dim), dtype=dtype)
            ),
            "self_attn.k_proj.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, feature_dim), dtype=dtype)
            ),
            "self_attn.v_proj.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, feature_dim), dtype=dtype)
            ),
            "self_attn.o_proj.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, feature_dim), dtype=dtype)
            ),
            "input_layernorm.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim), dtype=dtype)
            ),
            "mlp.gate_proj.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((ffn_dim, feature_dim), dtype=dtype)
            ),
            "mlp.up_proj.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((ffn_dim, feature_dim), dtype=dtype)
            ),
            "mlp.down_proj.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, ffn_dim), dtype=dtype)
            ),
            "post_attention_layernorm.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim), dtype=dtype)
            ),
        }
    )