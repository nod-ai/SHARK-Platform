# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum

import shortfin as sf
import shortfin.array as sfnp

from .io_struct import GenerateReqInput


class InferencePhase(Enum):
    # Tokenize prompt, negative prompt and get latents, timesteps, time ids, guidance scale as device arrays
    PREPARE = 1
    # Run CLIP to encode tokenized prompts into text embeddings
    ENCODE = 2
    # Run UNet to denoise the random sample
    DENOISE = 3
    # Run VAE to decode the denoised latents into an image.
    DECODE = 4
    # Postprocess VAE outputs.
    POSTPROCESS = 5


class InferenceExecRequest(sf.Message):
    """
    Generalized request passed for an individual phase of image generation.

    Used for individual image requests. Bundled as lists by the batcher for inference processes,
    and inputs joined for programs with bs>1.

    Inference execution processes are responsible for writing their outputs directly to the appropriate attributes here.
    """

    def __init__(
        self,
        phase: InferencePhase,
        prompt: str | None = None,
        neg_prompt: str | None = None,
        height: int | None = None,
        width: int | None = None,
        steps: int | None = None,
        guidance_scale: float | sfnp.device_array | None = None,
        seed: int | None = None,
        input_ids: list[list[int]] | None = None,
        sample: sfnp.device_array | None = None,
        prompt_embeds: sfnp.device_array | None = None,
        add_text_embeds: sfnp.device_array | None = None,
        timesteps: sfnp.device_array | None = None,
        time_ids: sfnp.device_array | None = None,
        denoised_latents: sfnp.device_array | None = None,
        image_array: sfnp.device_array | None = None,
    ):
        super().__init__()
        self.phase = phase

        # Phase inputs:
        # Prep phase.
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.height = height
        self.width = width
        self.seed = seed

        # Encode phase.
        # This is a list of sequenced positive and negative token ids and pooler token ids.
        self.input_ids = input_ids

        # Denoise phase.
        self.prompt_embeds = prompt_embeds
        self.add_text_embeds = add_text_embeds
        self.sample = sample
        self.steps = steps
        self.timesteps = timesteps
        self.time_ids = time_ids
        self.guidance_scale = guidance_scale

        # Decode phase.
        self.denoised_latents = denoised_latents

        # Postprocess.
        self.image_array = image_array

        self.result_image = None

        self.done = sf.VoidFuture()

        # Response control.
        # Move the result array to the host and sync to ensure data is
        # available.
        self.return_host_array: bool = False

    def reset(self, phase: InferencePhase):
        """Resets all per request state in preparation for an subsequent execution."""
        self.phase = phase
        self.done = sf.VoidFuture()
        self.return_host_array = True


class StrobeMessage(sf.Message):
    """Sent to strobe a queue with fake activity (generate a wakeup)."""

    ...
