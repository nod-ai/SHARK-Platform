# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Imports huggingface config and params into a Dataset.

This tool simply imports tensors and config with no transformation given a
config.json and a safetensors file. Once sharding configurations are worked
out, this should be replaced with a more general tool that can source from
either HF or an existing IRPA file and transform/save in one step.

Usage:
  python -m sharktank.models.punet.import_hf_dataset \
    --output-irpa-file ~/models/punet/punet_fp16.irpa \
    --config-json ~/models/stable-diffusion-xl-base-1.0/unet/config.json

The resulting dataset has all tensors as nested in the original model.
Properties are separated into a "meta" dict (for "_" prefixed props) and an
"hparams" dict.
"""

import json
from pathlib import Path
import sys

from ....types import *


def import_hf_config(config_json_path: Path, params_path: Path) -> Dataset:
    import safetensors

    with open(config_json_path, "rb") as f:
        config_json = json.load(f)
    # Separate meta parameters (prefixed with _) from hparams.
    meta_params = {k: v for k, v in config_json.items() if k.startswith("_")}
    hparams = {k: v for k, v in config_json.items() if not k.startswith("_")}

    with safetensors.safe_open(params_path, framework="pt", device="cpu") as st:
        tensors = [
            DefaultPrimitiveTensor(name=name, data=st.get_tensor(name))
            for name in st.keys()
        ]

    theta = Theta(tensors)
    props = {
        "meta": meta_params,
        "hparams": hparams,
    }
    return Dataset(props, theta)


def main(argv):
    from ....utils import cli

    parser = cli.create_parser()
    cli.add_output_dataset_options(parser)
    parser.add_argument(
        "--config-json", type=Path, required=True, help="Path to the config.json file"
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("diffusion_pytorch_model.fp16.safetensors"),
        help="Parameter file name, relative to config.json",
    )
    args = cli.parse(parser, args=argv)

    config_json_path: Path = args.config_json
    params_path: Path = args.params
    if not params_path.is_absolute():
        params_path = config_json_path.parent / params_path

    dataset = import_hf_config(config_json_path, params_path)
    dataset.save(args.output_irpa_file, io_report_callback=print)


if __name__ == "__main__":
    main(sys.argv[1:])
