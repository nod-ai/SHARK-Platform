# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shards an LLM dataset.

This is an ad-hoc transformation which operates on the layer structure of
weights of an LLM by converting the RHS of all eligible layers to a sharded
form.
"""
from ...transforms.dataset import MmtRHSShardingTransform
from ...types import *


def main(raw_args=None):
    from ...utils import cli

    parser = cli.create_parser()
    cli.add_input_dataset_options(parser)
    cli.add_output_dataset_options(parser)
    parser.add_argument(
        "--num-shards", type=int, required=True, help="Number of shards to split"
    )
    args = cli.parse(parser, args=raw_args)
    dataset = cli.get_input_dataset(args)

    tr = MmtRHSShardingTransform(
        r"^blk\.[0-9]+\.(attn_k|attn_q|attn_v|ffn_gate|ffn_up|ffn_down)\.weight$",
        num_shards=8,
    )
    dataset.transform(tr)
    dataset.save(args.output_irpa_file, io_report_callback=print)


if __name__ == "__main__":
    main()
