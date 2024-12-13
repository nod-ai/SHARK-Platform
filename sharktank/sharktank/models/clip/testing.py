# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ...layers.configs.llm_configs import ClipTextConfig
from ...types.theta import Theta
from .export import hugging_face_clip_text_model_to_theta
import torch


def make_clip_text_model_random_theta(config: ClipTextConfig) -> Theta:
    from transformers import CLIPTextConfig as HfCLIPTextConfig
    from transformers import CLIPTextModel as HfCLIPTextModel

    hf_config = config.to_hugging_face_clip_text_model_config()
    model = HfCLIPTextModel(hf_config)
    return hugging_face_clip_text_model_to_theta(model)


def make_random_input_token_sequences(
    batch_size: int, config: ClipTextConfig
) -> torch.LongTensor:
    sequence_lens = torch.randint(
        low=1, high=config.max_position_embeddings + 1, size=(batch_size,)
    )
    sequences = torch.full(
        size=(batch_size, config.max_position_embeddings),
        fill_value=config.eos_token_id,
        dtype=torch.long,
    )
    for batch_idx, l in enumerate(sequence_lens):
        sequences[batch_idx][0:l] = torch.randint(
            low=0, high=config.vocab_size - 1, size=(l,), dtype=torch.long
        )
    return sequences
