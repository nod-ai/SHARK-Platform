# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Objects transferred between components.

Portions adapted from API definitions originating in:

sglang: Copyright 2023-2024 SGLang Team, Licensed under the Apache License, Version 2.0
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import uuid


# Adapted from:
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/io_struct.py
@dataclass
class GenerateReqInput:
    # The input prompt. It can be a single prompt or a batch of prompts.
    prompt: Optional[Union[List[str], str]] = None
    # The input negative prompt. It can be a single prompt or a batch of prompts.
    neg_prompt: Optional[Union[List[str], str]] = None
    # Output image dimensions per prompt.
    height: Optional[Union[List[int], int]] = None
    width: Optional[Union[List[int], int]] = None
    # The number of inference steps; one int per prompt.
    steps: Optional[Union[List[int], int]] = None
    # The classifier-free-guidance scale for denoising; one float per prompt.
    guidance_scale: Optional[Union[List[float], float]] = None
    # The seed for random latents generation; one int per prompt.
    seed: Optional[Union[List[int], int]] = None
    # Token ids: only used in place of prompt.
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # Negative token ids: only used in place of negative prompt.
    neg_input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # Noisy latents, optionally specified for advanced workflows / inference comparisons
    latents: Optional[Union[List[sfnp.device_array], sfnp.device_array]] = None
    # The sampling parameters.
    sampling_params: Optional[Union[List[Dict], Dict]] = None
    # Output image format. Defaults to base64. One string ("PIL", "base64")
    output_type: Optional[List[str]] = None
    # The request id.
    rid: Optional[Union[List[str], str]] = None

    is_single: bool = True

    def post_init(self):
        if (self.prompt is None and self.input_ids is None) or (
            self.prompt is not None and self.input_ids is not None
        ):
            raise ValueError("Either text or input_ids should be provided.")

        prev_input_len = None
        for i in [self.prompt, self.neg_prompt, self.input_ids, self.neg_input_ids]:
            if isinstance(i, str):
                is_single = True
                self.num_output_images = 1
                continue
            elif not i:
                continue
            if not isinstance(i, list):
                raise ValueError("Text inputs should be strings or lists.")
            if prev_input_len and not (prev_input_len == len[i]):
                raise ValueError("Positive, Negative text inputs should be same length")
            prev_input_len = len(i)
        if not self.num_output_images:
            self.num_output_images = (
                len[self.prompt] if self.prompt is not None else len(self.input_ids)
            )
        if self.num_output_images > 1:
            is_single = False
        if self.sampling_params is None:
            self.sampling_params = [{}] * self.num_output_images
        elif not isinstance(self.sampling_params, list):
            self.sampling_params = [self.sampling_params] * self.num_output_images

        if self.rid is None:
            self.rid = [uuid.uuid4().hex for _ in range(num)]
        else:
            if not isinstance(self.rid, list):
                raise ValueError("The rid should be a list.")
        if self.output_type is None:
            self.output_type = ["base64"] * self.num_output_images
