# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from diffusers import UNet2DConditionModel


class ClassifierFreeGuidanceUnetModel(torch.nn.Module):
    def __init__(self, cond_model):
        super().__init__()
        self.cond_model = cond_model

    def forward(
        self,
        *,
        sample: torch.Tensor,
        timestep,
        prompt_embeds,
        text_embeds,
        time_ids,
        guidance_scale: torch.Tensor,
    ):
        added_cond_kwargs = {
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }
        latent_model_input = torch.cat([sample] * 2)
        noise_pred, *_ = self.cond_model.forward(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred


def main():
    from ....utils import cli

    parser = cli.create_parser()
    args = cli.parse(parser)

    device = "cuda:1"
    dtype = torch.float16

    cond_unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="unet",
        low_cpu_mem_usage=False,
        variant="fp16",
        torch_dtype=dtype,
    )
    print("Created model")
    cond_unet = cond_unet.to(device)
    print(cond_unet.conv_in.weight.dtype)
    mdl = ClassifierFreeGuidanceUnetModel(cond_unet)
    print("Moved to GPU")
    # print(mdl)

    # Run a step for debugging.
    torch.random.manual_seed(42)
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

    print("Calling forward")
    results = mdl.forward(
        sample=sample,
        timestep=timestep,
        prompt_embeds=prompt_embeds,
        text_embeds=text_embeds,
        time_ids=time_ids,
        guidance_scale=guidance_scale,
    )
    # print("1-step resutls:", results)


if __name__ == "__main__":
    main()
