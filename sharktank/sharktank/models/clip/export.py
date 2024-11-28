# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Union
import transformers
from transformers.models.clip.modeling_clip import (
    CLIPAttention as TransformersCLIPAttention,
    CLIPEncoderLayer as TransformersCLIPEncoderLayer,
    CLIPEncoder as TransformersCLIPEncoder,
)
from os import PathLike
import torch

from ...types.theta import Theta, Dataset, torch_module_to_theta
from ...types.tensors import DefaultPrimitiveTensor
from ...layers.configs import ClipTextConfig


def transformers_clip_attention_to_theta(model: TransformersCLIPAttention) -> Theta:
    return torch_module_to_theta(model)


def transformers_clip_encoder_layer_to_theta(model: TransformersCLIPEncoder) -> Theta:
    return torch_module_to_theta(model)


def transformers_clip_encoder_to_theta(model: TransformersCLIPEncoderLayer) -> Theta:
    return torch_module_to_theta(model)


def transformers_clip_text_model_to_theta(model: transformers.CLIPTextModel) -> Theta:
    return torch_module_to_theta(model)


def transformers_clip_text_model_to_dataset(
    model: transformers.CLIPTextModel,
) -> Dataset:
    config = ClipTextConfig.from_transformers_clip_text_config(model.config)
    properties = config.as_properties()
    theta = transformers_clip_text_model_to_theta(model)
    theta.rename_tensors_to_paths()
    return Dataset(properties, theta)


def export_clip_text_model_dataset_from_hugging_face(
    model_or_name_or_path: Union[str, PathLike, transformers.CLIPTextModel],
    output_path: Union[str, PathLike],
):
    if isinstance(model_or_name_or_path, transformers.CLIPTextModel):
        model = model_or_name_or_path
    else:
        model = transformers.CLIPTextModel.from_pretrained(model_or_name_or_path)
    dataset = transformers_clip_text_model_to_dataset(model)
    dataset.save(output_path)
