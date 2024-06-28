# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Specifications describing how block/layers of punet are sharded."""

from ...types.sharding import *


class ResnetBlock2DSplitOutputChannelsSharding(ThetaLayerSharding):
    """Shards the input channel and output channels of the convolutions."""

    def __init__(self, shard_count: int):
        super(Sharding).__init__()
        self.shard_count = shard_count

    def theta_sharding(self) -> ThetaSharding:
        result = ThetaSharding(
            {
                "norm1": GroupNormSplitChannelSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "conv1": Conv2DSplitOutputChannelSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "norm2": GroupNormSplitChannelSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "conv2": Conv2DSplitOutputChannelSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "time_emb_proj": LinearReplicatedInputSplitWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "conv_shortcut": Conv2DSplitOutputChannelSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
            }
        )
        return result
