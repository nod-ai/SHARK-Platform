# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch

from sharktank.models.punet.layers import ResnetBlock2D
from sharktank.types import *
from sharktank.models.punet.sharding import ResnetBlock2DSplitOutputChannelsSharding
from sharktank import ops


class ResnetBlockTest(unittest.TestCase):
    def testResnetBlock2DSplitInputAndOutputChannelsSharding(self):
        torch.set_default_dtype(torch.float32)
        batches = 2
        in_channels = 6
        out_channels = [12, 8]
        height = 17
        width = 19
        kernel_height = 5
        kernel_width = 5
        input_time_emb_shape = [batches, 8]
        norm_groups = 2
        eps = 0.01
        shard_count = 2
        theta = Theta(
            {
                "norm1.weight": DefaultPrimitiveTensor(
                    data=torch.rand(in_channels, dtype=torch.float32)
                ),
                "norm1.bias": DefaultPrimitiveTensor(
                    data=torch.rand(in_channels, dtype=torch.float32)
                ),
                "conv1.weight": DefaultPrimitiveTensor(
                    data=torch.rand(
                        out_channels[0],
                        in_channels,
                        kernel_height,
                        kernel_width,
                        dtype=torch.float32,
                    )
                ),
                "conv1.bias": DefaultPrimitiveTensor(
                    data=torch.rand(out_channels[0], dtype=torch.float32),
                ),
                "norm2.weight": DefaultPrimitiveTensor(
                    data=torch.rand(out_channels[0], dtype=torch.float32)
                ),
                "norm2.bias": DefaultPrimitiveTensor(
                    data=torch.rand(out_channels[0], dtype=torch.float32)
                ),
                "conv2.weight": DefaultPrimitiveTensor(
                    data=torch.rand(
                        out_channels[1],
                        out_channels[0],
                        kernel_height,
                        kernel_width,
                        dtype=torch.float32,
                    )
                ),
                "conv2.bias": DefaultPrimitiveTensor(
                    data=torch.rand(out_channels[1], dtype=torch.float32),
                ),
                "time_emb_proj.weight": DefaultPrimitiveTensor(
                    data=torch.rand(
                        out_channels[0], input_time_emb_shape[1], dtype=torch.float32
                    ),
                ),
                "time_emb_proj.bias": DefaultPrimitiveTensor(
                    data=torch.rand(out_channels[0], dtype=torch.float32),
                ),
                "conv_shortcut.weight": DefaultPrimitiveTensor(
                    data=torch.rand(
                        out_channels[1],
                        in_channels,
                        kernel_height,
                        kernel_width,
                        dtype=torch.float32,
                    )
                ),
                "conv_shortcut.bias": DefaultPrimitiveTensor(
                    data=torch.rand(out_channels[1], dtype=torch.float32),
                ),
            }
        )
        resnet_block = ResnetBlock2D(
            theta=theta,
            groups=norm_groups,
            eps=eps,
            non_linearity="relu",
            output_scale_factor=None,
            dropout=0.0,
            temb_channels=input_time_emb_shape[1],
            time_embedding_norm="default",
        )
        input_image = torch.rand(
            batches,
            in_channels,
            width,
            height,
        )
        input_time_emb = torch.rand(input_time_emb_shape)
        expected_result = resnet_block(input_image, input_time_emb)

        sharding_spec = ResnetBlock2DSplitOutputChannelsSharding(
            shard_count=shard_count
        )
        sharded_theta = ops.reshard(theta, sharding_spec)
        sharded_resnet_block = ResnetBlock2D(
            theta=sharded_theta,
            groups=norm_groups,
            eps=eps,
            non_linearity="relu",
            output_scale_factor=None,
            dropout=0.0,
            temb_channels=input_time_emb_shape[1],
            time_embedding_norm="default",
        )
        sharded_input_image = ops.reshard_split(input_image, dim=1, count=shard_count)
        sharded_input_time_emb = ops.replicate(input_time_emb, count=shard_count)
        sharded_result = sharded_resnet_block(
            sharded_input_image, sharded_input_time_emb
        )
        assert isinstance(sharded_result, SplitPrimitiveTensor)
        assert (
            sharded_result.shard_dim == 1 and sharded_result.shard_count == shard_count
        )
        actual_result = ops.unshard(sharded_result)
        torch.testing.assert_close(expected_result, actual_result)


if __name__ == "__main__":
    unittest.main()
