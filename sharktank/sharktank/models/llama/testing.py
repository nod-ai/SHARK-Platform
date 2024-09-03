# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List

import torch

from ...types.tensors import *
from ...types.theta import Theta


# Range of torch.rand() is [0,1)
# Range of torch.rand() * 2 - 1 is [-1, 1), includes negative values
def make_rand_torch(shape, dtype=torch.float32):
    return torch.rand(shape, dtype=dtype) * 2 - 1


def make_attention_block_theta(
    feature_dim: int,
    ffn_dim: int,
    dtype: torch.dtype | None = None,
) -> Theta:
    return Theta(
        {
            "attn_q.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, feature_dim), dtype=dtype)
            ),
            "attn_k.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, feature_dim), dtype=dtype)
            ),
            "attn_v.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, feature_dim), dtype=dtype)
            ),
            "attn_output.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, feature_dim), dtype=dtype)
            ),
            "attn_norm.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim), dtype=dtype)
            ),
            "ffn_gate.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((ffn_dim, feature_dim), dtype=dtype)
            ),
            "ffn_up.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((ffn_dim, feature_dim), dtype=dtype)
            ),
            "ffn_down.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, ffn_dim), dtype=dtype)
            ),
            "ffn_norm.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim), dtype=dtype)
            ),
        }
    )


def make_moe_block_theta(feature_dim=1024, ffn_dim=6144, num_experts=8) -> Theta:
    return Theta(
        {
            "blk.0.ffn_gate_inp.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, ffn_dim))
            ),
            "blk.0.ffn_norm.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((ffn_dim))
            ),
            "blk.0.layer_output_norm.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((ffn_dim))
            ),
            "blk.0.ffn_gate_exps.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((8, feature_dim * num_experts, ffn_dim))
            ),
            "blk.0.ffn_up_exps.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((8, feature_dim * num_experts, ffn_dim))
            ),
            "blk.0.ffn_down_exps.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((8, ffn_dim, feature_dim * num_experts))
            ),
        }
    )
