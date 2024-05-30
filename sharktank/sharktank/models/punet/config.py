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

from typing import List, Optional, Sequence, Tuple, Union

from dataclasses import dataclass
import inspect
import warnings

__all__ = [
    "HParams",
]


@dataclass
class HParams:
    act_fn: str = "silu"
    addition_embed_type: str = "text_time"
    addition_embed_type_num_heads: int = 64
    addition_time_embed_dim: Optional[int] = None
    block_out_channels: Sequence[int] = (320, 640, 1280, 1280)
    class_embed_type: Optional[str] = None
    class_embeddings_concat: bool = False
    center_input_sample: bool = False
    conv_in_kernel: int = (3,)
    down_block_types: Sequence[str] = ()
    downsample_padding: int = 1
    dropout: float = 0.0
    encoder_hid_dim: Optional[str] = None
    encoder_hid_dim_type: Optional[str] = None
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    in_channels: int = 4
    layers_per_block: Union[int, Tuple[int, ...]] = 2
    norm_eps: float = 1e-5
    norm_num_groups: int = 32
    projection_class_embeddings_input_dim: Optional[int] = (None,)
    resnet_out_scale_factor: float = 1.0
    resnet_time_scale_shift: str = "default"
    time_embedding_act_fn: Optional[str] = None
    time_embedding_dim: Optional[int] = None
    time_embedding_type: str = "positional"
    timestep_post_act: Optional[str] = None

    @property
    def downblock_layers_per_block(self) -> List[int]:
        if isinstance(self.layers_per_block, int):
            return [self.layers_per_block] * len(self.down_block_types)
        else:
            assert len(self.layers_per_block) == len(self.down_block_types)
            return self.layers_per_block

    def assert_default_values(self, attr_names: Sequence[str]):
        for name in attr_names:
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
