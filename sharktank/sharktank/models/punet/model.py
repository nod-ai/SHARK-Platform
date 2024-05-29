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

from ...layers import *
from ...types import *


@dataclass
class HParams:
    act_fn: str = "silu"
    addition_embed_type: str = "text_time"
    addition_embed_type_num_heads: int = 64
    addition_time_embed_dim: Optional[int] = None
    block_out_channels: Sequence[int] = (320, 640, 1280, 1280)
    class_embed_type: Optional[str] = None
    center_input_sample: bool = False
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    projection_class_embeddings_input_dim: Optional[int] = (None,)
    time_embedding_dim: Optional[int] = None
    time_embedding_type: str = "positional"
    timestep_post_act: Optional[str] = None

    def __post_init__(self):
        # We don't support the full parameterization of the diffusers model, so guard
        # parameters that we require to be their default. This is a tripwire in case
        # if we see a config that requires more support.
        require_default_attrs = [
            "addition_embed_type",
            "center_input_sample",
            "class_embed_type",
            "flip_sin_to_cos",
            "freq_shift",
            "time_embedding_dim",
            "time_embedding_type",
            "timestep_post_act",
        ]
        for name in require_default_attrs:
            actual = getattr(self, name)
            required = getattr(HParams, name)
            if actual != required:
                raise ValueError(
                    f"NYI: HParams.{name} != {required!r} (got {actual!r})"
                )

    @classmethod
    def from_dict(cls, d: dict):
        allowed = inspect.signature(cls).parameters
        declared_kwargs = {k: v for k, v in d.items() if k in allowed}
        extra_kwargs = [k for k in d.keys() if k not in allowed]
        if extra_kwargs:
            # TODO: Consider making this an error once bringup is done and we
            # handle everything.
            warnings.warn(f"Unhandled punet.HParams: {extra_kwargs}")
        return cls(**declared_kwargs)


ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}


class TimestepEmbedding(ThetaLayer):
    """Computes the embedding of projected timesteps.

    This consists of two linear layers with activation applied between.
    """

    def __init__(
        self, theta: Theta, in_channels: int, time_embed_dim: int, act_fn: str
    ):
        super().__init__(theta)
        self.in_channels = in_channels
        self.time_embed_dim = time_embed_dim
        try:
            self.act_fn = ACTIVATION_FUNCTIONS[act_fn]
        except KeyError as e:
            raise ValueError(f"Unknown activation function '{act_fn}'") from e

    def forward(self, sample):
        from ... import ops

        theta = self.theta
        weight_1 = theta.tensor("linear_1", "weight")
        bias_1 = theta.tensor("linear_1", "bias")
        weight_2 = theta.tensor("linear_2", "weight")
        bias_2 = theta.tensor("linear_2", "bias")
        h = ops.matmul(sample, weight_1)
        h = ops.elementwise(torch.add, h, bias_1)
        h = ops.elementwise(self.act_fn, h)
        h = ops.matmul(h, weight_2)
        h = ops.elementwise(torch.add, h, bias_2)
        return h


class Unet2DConditionModel(ThetaLayer):
    @classmethod
    def from_dataset(cls, ds: Dataset) -> "Unet2DConditionModel":
        hp = HParams.from_dict(ds.properties["hparams"])
        return cls(hp, ds.root_theta)

    def __init__(self, hp: HParams, theta: Theta):
        super().__init__(theta)
        self.hp = hp
        self._setup_timestep_embedding()
        self._setup_addition_embedding()

    def _setup_timestep_embedding(self):
        hp = self.hp
        assert hp.time_embedding_type == "positional", "NYI"
        time_embed_dim = hp.block_out_channels[0] * 2
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


class TimestepProjection(nn.Module):
    """Adapted from diffusers embeddings.get_timestep_embedding(), which claims:
        'This matches the implementation in Denoising Diffusion Probabilistic Models:
        Create sinusoidal timestep embeddings.'

    Args:
      embedding_dim: the dimension of the output.
      max_period: controls the minimum frequency of the
                  embeddings.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        max_period: int = 10000,
        downscale_freq_shift: float = 1.0,
        scale: float = 1.0,
    ):
        super().__init__()
        self.downscale_freq_shift = downscale_freq_shift
        self.embedding_dim = embedding_dim
        self.max_period = max_period
        self.scale = scale

    def forward(self, timesteps):
        """Args:
          timesteps: a 1-D Tensor of N indices, one per batch element.
                     These may be fractional.
        Returns:
          An [N x dim] Tensor of positional embeddings.
        """
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        embedding_dim = self.embedding_dim
        max_period = self.max_period
        downscale_freq_shift = self.downscale_freq_shift

        half_dim = embedding_dim // 2
        exponent = -math.log(max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - downscale_freq_shift)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        # scale embeddings
        emb = self.scale * emb

        # concat sine and cosine embeddings
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # zero pad
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


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

    ds = cli.get_input_dataset(args)
    cond_unet = Unet2DConditionModel.from_dataset(ds)
    mdl = ClassifierFreeGuidanceUnetModel(cond_unet)
    print(f"Model hparams: {cond_unet.hp}")

    # Run a step for debugging.
    dtype = torch.float16
    max_length = 64
    height = 1024
    width = 1024
    bs = 1
    sample = torch.rand(bs, 4, height // 8, width // 8, dtype=dtype)
    timestep = torch.zeros(1, dtype=torch.int32)
    prompt_embeds = torch.rand(2 * bs, max_length, 2048, dtype=dtype)
    text_embeds = torch.rand(2 * bs, 1280, dtype=dtype)
    time_ids = torch.zeros(2 * bs, 6, dtype=dtype)
    guidance_scale = torch.tensor([7.5], dtype=dtype)

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
