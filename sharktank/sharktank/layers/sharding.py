# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Specifications describing how layers are sharded."""

from ..types.sharding import *


class PagedLlamaAttentionBlockSharding(ThetaLayerSharding):
    def __init__(self, shard_count: int):
        super().__init__()
        self.shard_count = shard_count

    def theta_sharding(self) -> ThetaSharding:
        return ThetaSharding(
            {
                # The size of this is the token embedding length, which is not a memory
                # space concern if replicated even for all attention blocks.
                "attn_norm": RmsNormReplicatedSharding(
                    self.shard_count
                ).theta_sharding(),
                "attn_q": LinearSplitParallelWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "attn_k": LinearSplitParallelWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "attn_v": LinearSplitParallelWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "attn_output": LinearSplitReductionDimSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
            }
        )
