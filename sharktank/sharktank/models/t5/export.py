# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
from typing import Optional, Union
from pathlib import Path
import torch
from copy import copy

from .t5 import T5Config, T5Encoder
from ...types import Dataset
from ...transforms.dataset import set_float_dtype
from iree.turbine.aot import FxProgramsBuilder, export

__all__ = [
    "export_encoder_mlir",
    "export_encoder_iree_parameters",
    "prune_decoder_parameters",
]


def export_encoder_mlir(
    model: Union[T5Encoder, Path, str],
    batch_sizes: list[int],
    mlir_output_path: str,
):
    """
    Args:
      model: either the torch module or path to GGUF/IRPA.
    """
    if isinstance(model, (Path, str)):
        dataset = Dataset.load(model)
        config = T5Config.from_gguf_properties(
            dataset.properties,
            # TODO: add this property to our HuggingFace-to-GGUF conversion script.
            # We currently use llama.cpp's converter and it can not make a distinction
            # between T5 V1 and V1.1.
            # V1 uses ReLU and V1.1 uses gated GeLU.
            feed_forward_proj="gated-gelu",
        )
        model = T5Encoder(theta=dataset.root_theta, config=config)

    fxb = FxProgramsBuilder(model)

    for batch_size in batch_sizes:
        sample_inputs = model.sample_inputs(batch_size)

        context_length_dim_idx = 1
        assert (
            sample_inputs["input_ids"].shape[context_length_dim_idx]
            % config.context_length_padding_block_size
            == 0
        )
        context_length_block_dim_max = (
            sample_inputs["input_ids"].shape[context_length_dim_idx]
            // config.context_length_padding_block_size
        )
        context_length_block_dim = torch.export.Dim(
            "block", max=context_length_block_dim_max
        )
        context_length_dim = (
            config.context_length_padding_block_size * context_length_block_dim
        )
        dynamic_shapes = {"input_ids": {context_length_dim_idx: context_length_dim}}

        @fxb.export_program(
            name=f"forward_bs{batch_size}",
            args=tuple(sample_inputs.values()),
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
        def _(
            model,
            input_ids,
        ):
            return model(input_ids)

    output = export(fxb, import_symbolic_shape_expressions=True)
    output.save_mlir(mlir_output_path)


def prune_decoder_parameters(dataset: Dataset):
    # Remove decoder tensors/parameters if present.
    try:
        del dataset.root_theta.tree["dec"]
    except KeyError:
        pass
    try:
        del dataset.properties["t5.decoder_start_token_id"]
    except KeyError:
        pass


def export_encoder_iree_parameters(
    model_path_or_dataset: str | Dataset,
    output_path: str,
    dtype: Optional[torch.dtype] = None,
):
    if isinstance(model_path_or_dataset, Dataset):
        dataset = copy(model_path_or_dataset)
    else:
        dataset = Dataset.load(model_path_or_dataset)
    if dtype:
        dataset.root_theta = dataset.root_theta.transform(
            functools.partial(set_float_dtype, dtype=dtype)
        )
    prune_decoder_parameters(dataset)
    dataset.save(output_path)
