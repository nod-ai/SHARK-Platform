# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch.nn as nn

from sharktank.layers import *
from sharktank import ops
from ...types import *

from .config import *
from .layers import *
from sharktank.models.punet.layers import UpDownBlock2D, GroupNormLayer
from typing import Optional


class VaeDecoderModel(ThetaLayer):
    @classmethod
    def from_dataset(cls, ds: Dataset) -> "VaeDecoderModel":
        hp = HParams.from_dict(ds.properties["hparams"])
        return cls(hp, ds.root_theta)

    def __init__(self, hp: HParams, theta: Theta):
        super().__init__(theta)
        self.hp = hp

        # input conv
        self.post_quant_conv = Conv2DLayer(theta("post_quant_conv"), padding=(0, 0))
        self.conv_in = Conv2DLayer(theta("decoder")("conv_in"), padding=(1, 1))
        # Mid
        self.mid_block = self._create_mid_block(theta("decoder")("mid_block"))
        # up
        self.up_blocks = nn.ModuleList([])
        self.upscale_dtype = theta("decoder")("up_blocks")(0)("resnets")(0)("conv1")(
            "weight"
        ).dtype
        for i, up_block_name in enumerate(hp.up_block_types):
            up_block_theta = theta("decoder")("up_blocks")(i)
            is_final_block = i == len(hp.block_out_channels) - 1
            self.up_blocks.append(
                self._create_up_block(
                    up_block_theta,
                    up_block_name,
                    is_final_block=is_final_block,
                )
            )
        # TODO add spatial norm type support
        self.conv_norm_out = GroupNormLayer(
            theta("decoder")("conv_norm_out"), num_groups=hp.norm_num_groups, eps=1e-6
        )

        self.conv_act = nn.SiLU()
        self.conv_out = Conv2DLayer(theta("decoder")("conv_out"), padding=(1, 1))

    def forward(
        self, sample: torch.Tensor, latent_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        The forward method of the 'Decoder' class
        Args:
            sample ('torch.Tensor') input latents of shape (batch_size, num_channels, height, width)

        """
        self.trace_goldens(
            "inputs",
            {
                "sample": sample,
                "latent_embeds": latent_embeds,
            },
        )
        sample = 1 / self.hp.scaling_factor * sample

        sample = self.post_quant_conv(sample)
        sample = self.conv_in(sample)
        self.trace_golden("conv_in", sample)
        # TODO add training and gradient checkpointing support
        sample = self.mid_block(sample, latent_embeds)
        self.trace_golden("mid_block", sample)

        sample = sample.to(self.upscale_dtype)
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)

        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        sample = (sample / 2 + 0.5).clamp(0, 1)
        return sample

    def _create_mid_block(self, mid_block_theta: Theta) -> nn.Module:
        hp = self.hp
        return UNetMidBlock2D(
            mid_block_theta,
            temb_channels=None,
            dropout=0.0,
            num_layers=hp.layers_per_block,
            resnet_eps=1e-6,
            resnet_act_fn="swish",
            resnet_groups=hp.norm_num_groups,
            attn_groups=hp.norm_num_groups,
            resnet_pre_norm=True,
            add_attention=True,
            attention_head_dim=hp.block_out_channels[-1],
            output_scale_factor=1.0,
            resnet_time_scale_shift="default",
        )

    def _create_up_block(
        self, up_block_theta: Theta, type_name: str, is_final_block: bool
    ) -> nn.Module:
        hp = self.hp
        if type_name == "UpDecoderBlock2D":
            return UpDecoderBlock2D(
                up_block_theta,
                num_layers=hp.layers_per_block + 1,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=hp.act_fn,
                resnet_groups=hp.norm_num_groups,
                resnet_time_scale_shift="default",
                temb_channels=None,
                dropout=0.0,
                resnet_out_scale_factor=None,
            )
