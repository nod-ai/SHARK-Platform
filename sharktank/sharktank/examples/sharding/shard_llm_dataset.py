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
from ...models.llama.sharding import shard_theta
from ...layers import LlamaHParams, LlamaModelConfig
from ...types import *


def main(raw_args=None):
    from ...utils import cli

    parser = cli.create_parser()
    cli.add_input_dataset_options(parser)
    cli.add_output_dataset_options(parser)
    parser.add_argument(
        "--tensor-parallelism-size",
        type=int,
        required=True,
        help="Number of shards to split",
    )
    args = cli.parse(parser, args=raw_args)
    dataset = cli.get_input_dataset(args)

    hp = LlamaHParams.from_gguf_props(dataset.properties)
    llama_config = LlamaModelConfig(
        hp, tensor_parallelism_size=args.tensor_parallelism_size
    )
    sharded_theta = shard_theta(dataset.root_theta, llama_config)
    sharded_theta.rename_tensors_to_paths()
    dataset.root_theta = sharded_theta
    dataset.properties["tensor_parallelism_size"] = args.tensor_parallelism_size
    dataset.save(args.output_irpa_file, io_report_callback=print)


if __name__ == "__main__":
    main()
