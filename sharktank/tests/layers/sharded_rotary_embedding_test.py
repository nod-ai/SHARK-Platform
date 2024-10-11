# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import torch

from sharktank.layers import RotaryEmbeddingLayer
from sharktank import ops
from sharktank.types import (
    ShardedTensor,
    SplitPrimitiveTensor,
    unbox_tensor,
)

import unittest
from typing import List, Optional
import os


def test_sharded_rotary_table():
    bs = 4
    rope_dims = 16
    heads = 8
    max_seqlen = 128
    rope_freq_base = None

    # First we setup and get the default rotary embedding layer
    xq = torch.rand((bs, max_seqlen, heads, rope_dims), dtype=torch.float)
    xk = torch.rand((bs, max_seqlen, heads, rope_dims), dtype=torch.float)
    default_layer = RotaryEmbeddingLayer(
        rope_dimension_count=rope_dims,
        max_seqlen=max_seqlen,
        rope_freq_base=rope_freq_base,
    )
    oq, ok = default_layer(xq=xq, xk=xk, start_index=0)

    # Then we can shard the same inputs and layer
    xq = SplitPrimitiveTensor(ts=xq, shard_dim=2, shard_count=4)
    xk = SplitPrimitiveTensor(ts=xk, shard_dim=2, shard_count=4)
    shard_layer = RotaryEmbeddingLayer(
        rope_dimension_count=rope_dims,
        max_seqlen=max_seqlen,
        rope_freq_base=rope_freq_base,
        tensor_parallelism_size=4,
    )
    sq, sk = shard_layer(xq=xq, xk=xk, start_index=0)

    # Gathering and unboxing should yield the same results
    sq = ops.all_gather(sq)
    sk = ops.all_gather(sk)
    sq = unbox_tensor(sq.shards[0])
    sk = unbox_tensor(sk.shards[0])

    torch.testing.assert_close(sq, oq)
    torch.testing.assert_close(sk, ok)
