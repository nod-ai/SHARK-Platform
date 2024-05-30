# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Optional, Sequence, Tuple

import math

import torch
import torch.nn as nn

from ... import ops
from ...layers import *
from ...types import *
from .config import *


__all__ = [
    "ACTIVATION_FUNCTIONS",
    "Conv2DLayer",
    "CrossAttnDownBlock2D",
    "DownBlock2D",
    "TimestepEmbedding",
    "TimestepProjection",
]

################################################################################
# Down blocks.
################################################################################


class DownBlock2D(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        *,
        num_layers: int,
        add_downsample: bool,
        resnet_eps: float,
        resnet_act_fn: str,
        resnet_groups: int,
        resnet_out_scale_factor: Optional[float],
        resnet_time_scale_shift: str,
        temb_channels: int,
        dropout: float,
        downsample_padding: int,
    ):
        super().__init__(theta)
        resnets = []
        for i in range(num_layers):
            resnets.append(
                ResnetBlock2D(
                    theta("resnets", i),
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=resnet_out_scale_factor,
                    time_embedding_norm=resnet_time_scale_shift,
                    temb_channels=temb_channels,
                    dropout=dropout,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        assert not add_downsample, "NYI: DownBlock2D add_downsample"

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None):
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
        output_states = output_states + (hidden_states,)
        return hidden_states, output_states


class CrossAttnDownBlock2D(ThetaLayer):
    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None):
        raise NotImplementedError


################################################################################
# Resnet block.
################################################################################


class ResnetBlock2D(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        groups: int,
        eps: float,
        non_linearity: str,
        output_scale_factor: Optional[float],
        dropout: float,
        temb_channels: int,
        time_embedding_norm: str,
    ):
        super().__init__(theta)
        assert dropout == 0.0, "ResnetBlock2D currently does not support dropout"
        self.norm1 = GroupNormLayer(theta("norm1"), num_groups=groups, eps=eps)
        self.conv1 = Conv2DLayer(theta("conv1"), padding=(1, 1))
        self.norm2 = GroupNormLayer(theta("norm2"), num_groups=groups, eps=eps)
        self.conv2 = Conv2DLayer(theta("conv2"), padding=(1, 1))
        self.nonlinearity = ACTIVATION_FUNCTIONS[non_linearity]
        self.output_scale_factor = output_scale_factor

        self.time_emb_proj = None
        if temb_channels is not None:
            assert (
                time_embedding_norm == "default"
            ), f"NYI: ResnetBlock2D(time_embedding_norm={time_embedding_norm})"
            self.time_emb_proj = LinearLayer(theta("time_emb_proj"))

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = ops.elementwise(self.nonlinearity, hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            temb = ops.elementwise(self.nonlinearity, temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]
            hidden_states = ops.elementwise(torch.add, hidden_states, temb)

        hidden_states = self.norm2(hidden_states)
        hidden_states = ops.elementwise(self.nonlinearity, hidden_states)
        hidden_states = self.conv2(hidden_states)

        output_tensor = input_tensor + hidden_states
        if self.output_scale_factor is not None:
            output_tensor = output_tensor / self.output_scale_factor
        return output_tensor


class GroupNormLayer(ThetaLayer):
    def __init__(self, theta: Theta, num_groups: int, eps: float, affine: bool = True):
        super().__init__(theta)
        assert affine, "NYI: GroupNormLayer(affine=False)"
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.theta.tensor("weight")
        bias = self.theta.tensor("bias")
        return ops.group_norm_affine(
            input, weight, bias, num_groups=self.num_groups, eps=self.eps
        )


################################################################################
# Utility layers.
################################################################################

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
            raise AssertionError(f"Unknown activation function '{act_fn}'") from e

    def forward(self, sample):
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


################################################################################
# Layers that need to be made common once stable.
################################################################################


class Conv2DLayer(ThetaLayer):
    """Theta based conv2d layer. This assumes weight/bias naming as per the nn.Conv2D
    module ("weight", "bias").

    """

    def __init__(self, theta: Theta, padding: Optional[Tuple[int, int]] = None):
        super().__init__(theta)
        assert padding is None or len(padding) == 2
        self.padding = padding
        self.stride = 1
        self.dilation = 1
        self.groups = 1

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.theta.tensor("weight")
        if "bias" in self.theta.keys:
            bias = self.theta.tensor("bias")
        else:
            bias = None
        return ops.conv2d(
            input,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
