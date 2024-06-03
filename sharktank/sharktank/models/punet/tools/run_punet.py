# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from ..model import Unet2DConditionModel, ClassifierFreeGuidanceUnetModel


def main():
    from ....utils import cli

    parser = cli.create_parser()
    cli.add_input_dataset_options(parser)
    parser.add_argument("--device", default="cuda:0", help="Torch device to run on")
    parser.add_argument("--dtype", default="float16", help="DType to run in")
    args = cli.parse(parser)

    device = args.device
    dtype = getattr(torch, args.dtype)

    ds = cli.get_input_dataset(args)
    ds.to(device=device)

    cond_unet = Unet2DConditionModel.from_dataset(ds)
    mdl = ClassifierFreeGuidanceUnetModel(cond_unet)

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
