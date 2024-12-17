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
    --config-json ~/models/stable-diffusion-xl-base-1.0/unet/config.json \
    --params diffusion_pytorch_model.fp16.safetensors

The resulting dataset has all tensors as nested in the original model.
Properties are separated into a "meta" dict (for "_" prefixed props) and an
"hparams" dict.
"""

from typing import Optional
from os import PathLike
import json
from pathlib import Path
import sys
import logging

from ..types import *

logger = logging.getLogger(__name__)


def import_hf_dataset(
    config_json_path: PathLike,
    param_paths: list[PathLike],
    output_irpa_file: Optional[PathLike] = None,
) -> Optional[Dataset]:
    import safetensors

    with open(config_json_path, "rb") as f:
        config_json = json.load(f)
    # Separate meta parameters (prefixed with _) from hparams.
    meta_params = {k: v for k, v in config_json.items() if k.startswith("_")}
    hparams = {k: v for k, v in config_json.items() if not k.startswith("_")}

    for params_path in param_paths:
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
    dataset = Dataset(props, theta)

    if output_irpa_file is None:
        return dataset

    dataset.save(output_irpa_file, io_report_callback=logger.info)


def main(argv: list[str]):
    from ..utils import cli

    parser = cli.create_parser()
    cli.add_output_dataset_options(parser)
    parser.add_argument(
        "--config-json", type=Path, required=True, help="Path to the config.json file"
    )
    parser.add_argument(
        "--params",
        type=Path,
        nargs="+",
        default=Path("diffusion_pytorch_model.fp16.safetensors"),
        help="Parameter file name(s), relative to config.json",
    )
    args = cli.parse(parser, args=argv)

    config_json_path: Path = args.config_json
    param_paths: list[Path] = args.params
    param_paths = [
        path if path.is_absolute() else config_json_path.parent / path
        for path in param_paths
    ]

    import_hf_dataset(
        config_json_path, param_paths, output_irpa_file=args.output_irpa_file
    )


if __name__ == "__main__":
    main(sys.argv[1:])
