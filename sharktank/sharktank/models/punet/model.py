# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Significant portions of this implementation were derived from diffusers,
# licensed under Apache2: https://github.com/huggingface/diffusers
# While much was a simple reverse engineering of the config.json and parameters,
# code was taken where appropriate.
from typing import Optional, Sequence, Tuple

from dataclasses import dataclass
import inspect
import math
import warnings

import torch
import torch.nn as nn

from ... import ops
from ...layers import *
from ...types import *

from .config import *
from .layers import *


class Unet2DConditionModel(ThetaLayer):
    @classmethod
    def from_dataset(cls, ds: Dataset) -> "Unet2DConditionModel":
        hp = HParams.from_dict(ds.properties["hparams"])
        return cls(hp, ds.root_theta)

    def __init__(self, hp: HParams, theta: Theta):
        super().__init__(theta)
        self.hp = hp
        # We don't support the full parameterization of the diffusers model, so guard
        # parameters that we require to be their default. This is a tripwire in case
        # if we see a config that requires more support.
        hp.assert_default_values(
            [
                "addition_embed_type",
                "center_input_sample",
                "class_embed_type",
                "class_embeddings_concat",
                "encoder_hid_dim",
                "encoder_hid_dim_type",
                "flip_sin_to_cos",
                "freq_shift",
                "time_embedding_act_fn",
                "time_embedding_dim",
                "time_embedding_type",
                "timestep_post_act",
            ]
        )
        self._setup_timestep_embedding()
        self._setup_addition_embedding()

        # Input convolution.
        conv_in_padding = (hp.conv_in_kernel - 1) // 2
        self.conv_in = Conv2DLayer(
            theta("conv_in"), padding=(conv_in_padding, conv_in_padding)
        )

        # Down/up blocks.
        output_channel = hp.block_out_channels[0]
        self.down_blocks = nn.ModuleList([])
        for i, down_block_name in enumerate(hp.down_block_types):
            input_channel = output_channel
            output_channel = hp.block_out_channels[i]
            down_block_theta = theta("down_blocks", i)
            is_final_block = i == len(hp.block_out_channels) - 1
            self.down_blocks.append(
                self._create_down_block(
                    i,
                    down_block_theta,
                    down_block_name,
                    is_final_block=is_final_block,
                )
            )

    def forward(
        self,
        *,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        text_embeds: torch.Tensor,
        time_ids: torch.Tensor,
    ):
        """
        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            text_embed: Additional embedding.
            time_ids: Additional embedding.
        """
        # Invariants.
        torch._check(len(timestep.shape) == 1 and timestep.shape[0] == 1)
        # TODO: Verify on the fly upsampling is not needed (num_upsamplers != 0).
        act_dtype = sample.dtype
        bs, *_ = sample.shape

        # 0. Center input if necessary.
        assert not self.hp.center_input_sample, "NYI: Center input sample"

        # 1. Embeddings.
        # 1a. Time embedding.
        # Broadcast the timestep to the batch size ([1] -> [bs]), apply projection
        # and cast.
        t_emb = self.time_proj(timestep.expand(bs)).to(dtype=act_dtype)
        emb = self.time_embedding(t_emb)

        # 1b. Aug embedding of text_embeds, time_ids
        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1).to(emb.dtype)
        aug_embed = self.add_embedding(add_embeds)
        emb = emb + aug_embed

        # 2. Pre-process.
        sample = self.conv_in(sample)

        # 3. Down.
        downblock_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            downblock_res_samples += res_samples

        # 4. Mid.

        # 5. Up.

        # 6. Post-process.

    def _create_down_block(
        self, i: int, down_block_theta: Theta, type_name: str, is_final_block: bool
    ) -> nn.Module:
        hp = self.hp
        if type_name == "DownBlock2D":
            return DownBlock2D(
                down_block_theta,
                num_layers=hp.downblock_layers_per_block[i],
                add_downsample=is_final_block,
                resnet_eps=hp.norm_eps,
                resnet_act_fn=hp.act_fn,
                resnet_groups=hp.norm_num_groups,
                resnet_out_scale_factor=None
                if hp.resnet_out_scale_factor == 1.0
                else hp.resnet_out_scale_factor,
                resnet_time_scale_shift=hp.resnet_time_scale_shift,
                dropout=hp.dropout,
                downsample_padding=hp.downsample_padding,
                temb_channels=self.time_embed_dim,
            )
        elif type_name == "CrossAttnDownBlock2D":
            return CrossAttnDownBlock2D(down_block_theta)
        raise ValueError(f"Unhandled down_block_type: {type_name}")

    def _setup_timestep_embedding(self):
        hp = self.hp
        assert hp.time_embedding_type == "positional", "NYI"
        self.time_embed_dim = time_embed_dim = hp.block_out_channels[0] * 2
        timestep_input_dim = hp.block_out_channels[0]
        self.time_proj = TimestepProjection(hp.block_out_channels[0])
        self.time_embedding = TimestepEmbedding(
            self.theta("time_embedding"),
            timestep_input_dim,
            time_embed_dim,
            act_fn=hp.act_fn,
        )

    def _setup_addition_embedding(self):
        hp = self.hp
        assert hp.addition_embed_type == "text_time", "NYI"
        self.add_time_proj = TimestepProjection(
            hp.addition_time_embed_dim, downscale_freq_shift=hp.freq_shift
        )
        self.add_embedding = TimestepEmbedding(
            self.theta("add_embedding"),
            in_channels=hp.projection_class_embeddings_input_dim,
            time_embed_dim=hp.time_embedding_dim,
            act_fn=hp.act_fn,
        )


class ClassifierFreeGuidanceUnetModel(torch.nn.Module):
    def __init__(self, cond_model: Unet2DConditionModel):
        super().__init__()
        self.cond_model = cond_model

    def forward(
        self, *, sample: torch.Tensor, guidance_scale: torch.Tensor, **cond_kwargs
    ):
        latent_model_input = torch.cat([sample] * 2)
        noise_pred = self.cond_model.forward(sample=latent_model_input, **cond_kwargs)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred


def main():
    from ...utils import cli

    parser = cli.create_parser()
    cli.add_input_dataset_options(parser)
    args = cli.parse(parser)

    device = "cuda:0"
    ds = cli.get_input_dataset(args)
    ds.to(device=device)

    cond_unet = Unet2DConditionModel.from_dataset(ds)
    mdl = ClassifierFreeGuidanceUnetModel(cond_unet)
    print(f"Model hparams: {cond_unet.hp}")

    # Run a step for debugging.
    dtype = torch.float16
    max_length = 64
    height = 1024
    width = 1024
    bs = 1
    sample = torch.rand(bs, 4, height // 8, width // 8, dtype=dtype).to(device)
    timestep = torch.zeros(1, dtype=torch.int32).to(device)
    prompt_embeds = torch.rand(2 * bs, max_length, 2048, dtype=dtype).to(device)
    text_embeds = torch.rand(2 * bs, 1280, dtype=dtype).to(device)
    time_ids = torch.zeros(2 * bs, 6, dtype=dtype).to(device)
    guidance_scale = torch.tensor([7.5], dtype=dtype).to(device)

    results = mdl.forward(
        sample=sample,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        text_embeds=text_embeds,
        time_ids=time_ids,
        guidance_scale=guidance_scale,
    )
    print("1-step resutls:", results)


if __name__ == "__main__":
    main()
