# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import re

import numpy as np
import torch

from ....layers import configs
from ..llama import LlamaModelConfig
from ..sharding import shard_theta


def main():
    from ....utils import cli

    parser = cli.create_parser()
    cli.add_input_dataset_options(parser)
    parser.add_argument(
        "--output-file", type=Path, help="Save the dataset to an IRPA file"
    )
    parser.add_argument(
        "--shard_count",
        required=True,
        type=int,
        help="Level of parallelism in sharding",
    )
    args = cli.parse(parser)
    dataset = cli.get_input_dataset(args)

    if args.output_file is None:
        raise RuntimeError(f"Need file destination for IRPA file")

    if args.shard_count < 2:
        raise RuntimeError(f"Expect sharding greater than 1 found {args.shard_count}")

    hp = configs.LlamaHParams.from_gguf_props(dataset.properties)
    llama_config = LlamaModelConfig(hp)
    llama_config.kv_cache_type = "paged"
    llama_config.tensor_parallelism_size = args.shard_count
    dataset.root_theta = shard_theta(dataset.root_theta, llama_config)

    def report(s):
        print(f"Save: {s}")

    print(f"Saving to: {args.output_file}")
    dataset.save(args.output_file, io_report_callback=report)


if __name__ == "__main__":
    main()
