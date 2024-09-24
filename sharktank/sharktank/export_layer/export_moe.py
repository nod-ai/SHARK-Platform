# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from iree.turbine.aot import *
from sharktank.models.llama.testing import make_moe_block_theta, make_rand_torch
from sharktank.layers.mixture_of_experts_block import PreGatherMoeBlock
from ..utils import cli


def main():
    parser = cli.create_parser()
    parser.add_argument(
        "--output-mlir",
        help="Output file path for exported MLIR file",
        default="/tmp/batch_llama_v1.mlir",
    )
    parser.add_argument(
        "--batch-size",
        "-bs",
        help="Batch size to generate, e.g. `4` or `2`",
        type=lambda arg: int(arg),
        default="2",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        help="Include verbose logging",
        action="store_true",
    )
    parser.add_argument(
        "--strict",
        help="Enables strictness during export",
        action="store_true",
    )
    parser.add_argument(
        "--use-grok",
        help="Enable to export Grok model's version of MOE block",
        action="store_true",
    )

    args = cli.parse(parser)

    bs = args.batch_size

    model = PreGatherMoeBlock(
        theta=make_moe_block_theta()("blk.0"),
        expert_count=8,
        expert_used_count=2,
        rms_epsilon=1e-5,
        use_grok=args.use_grok,
    )
    fxb = FxProgramsBuilder(model)
    input = make_rand_torch((bs, 32, 6144))

    @fxb.export_program(name="prefill_moe", args=(input,))
    def _(model, input: torch.Tensor) -> torch.Tensor:
        return model(input)

    input = make_rand_torch((bs, 1, 6144))

    @fxb.export_program(name="decode_moe", args=(input,))
    def _(model, input: torch.Tensor) -> torch.Tensor:
        return model(input)

    if args.verbose:
        for name, ep in fxb.programs.items():
            print(f"EXPORT {name}:\n{ep}")

    print("Exporting")
    output = export(fxb)
    print(f"Saving to '{args.output_mlir}'")
    output.save_mlir(args.output_mlir)


if __name__ == "__main__":
    main()
