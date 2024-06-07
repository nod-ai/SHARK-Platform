# Copyright 2024 Advanced Micro Devices, Inc
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
from typing import Optional

import json
from pathlib import Path
import safetensors
import torch

from ....types import *


def _load_json(p: Path):
    print(f"Loading {p}")
    with open(p, "rb") as f:
        return json.load(f)


def _get_dataset_props(config_json_struct) -> dict:
    # Separate meta parameters (prefixed with _) from hparams.
    meta_params = {k: v for k, v in config_json_struct.items() if k.startswith("_")}
    hparams = {k: v for k, v in config_json_struct.items() if not k.startswith("_")}
    return {
        "meta": meta_params,
        "hparams": hparams,
    }


def _load_theta(st_source) -> Theta:
    tensors = [
        DefaultPrimitiveTensor(name=name, data=st_source.get_tensor(name))
        for name in st_source.keys()
    ]
    return Theta(tensors)


def apply_per_layer_quant(theta: Theta, qp):
    # The config file has tensors as 1d and a _shape suffixed array with the
    # concrete shape.
    def _get_json_tensor(name: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
        data_1d = qp.get(name)
        if data_1d is None:
            return None
        shape = qp[f"{name}_shape"]
        return torch.tensor(data_1d, dtype=dtype).reshape(shape)

    input_scale = _get_json_tensor("input_scale", torch.float32)
    input_zp = _get_json_tensor("input_zp", torch.int32)
    weight_scale = _get_json_tensor("weight_scale", torch.float32)
    weight_zp = _get_json_tensor("weight_zp", torch.int32)
    assert (
        weight_scale is not None and weight_zp is not None
    ), f"Could not find weight scale (in {qp.keys()})"

    weight = theta.tensor("weight")
    # TODO: There is ambiguity with respect to the axis of quantization and the
    # shape of the scale. I think the intent was that the scale should have a
    # broadcast ready shape, but they are all 1d. This model only does axis-0
    # quantization, so we just assert that matches for now.
    assert weight.shape[0] == weight_scale.shape[0], "Mismatched quantization axis"

    # There is an implicit assumption that the weight is asym (uint8) quantized.
    # Our quantizer uses scale/offset style:
    #  round(t - offset) * scale)
    #  round(t * scale - offset * scale)
    # Whereas the params here are scale/zero-point:
    #  round(t / scale + zp)
    # We haven't used our infra for much asymmetric quant and may want to adapt
    # zero point handling if rounding/precision is an issue.
    quant_scale = 1.0 / weight_scale
    quant_rscale = weight_scale
    quant_offset = torch.neg(weight_zp * quant_rscale)
    quantizer = StaticScaledQuantizer(
        scale=quant_scale,
        reciprocal_scale=quant_rscale,
        axis=0,
        offset=quant_offset,
        dtype=torch.uint8,
    )
    weight_quant = quantizer.quantize(weight)
    layout = weight_quant.unpack()
    weight_dequant = layout.dequant()
    print(f"ORIG:\n{weight.as_torch()[0]}")
    print(f"DEQUANT:\n{weight_dequant[0]}")


def main():
    from ....utils import cli

    parser = cli.create_parser()
    cli.add_output_dataset_options(parser)
    parser.add_argument(
        "--config-json", type=Path, required=True, help="Path to the config.json file"
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("params.safetensors"),
        help="Parameter file name, relative to config.json",
    )
    parser.add_argument(
        "--quant-params",
        type=Path,
        default=Path("quant_param.json"),
        help="Quantization parameters",
    )
    args = cli.parse(parser)

    config_json_path: Path = args.config_json
    params_path: Path = args.params
    quant_params_path: Path = args.quant_params
    if not params_path.is_absolute():
        params_path = config_json_path.parent / params_path
    if not quant_params_path.is_absolute():
        quant_params_path = config_json_path.parent / quant_params_path

    # Construct the pre-transform dataset.
    dataset_props = _get_dataset_props(_load_json(config_json_path))
    quant_params_struct = _load_json(quant_params_path)
    with safetensors.safe_open(params_path, framework="pt", device="cpu") as st:
        theta = _load_theta(st)
    ds = Dataset(dataset_props, theta)

    # The quant_params_struct has quantization parameter structs keyed by full
    # layer name. We process each of these in turn to produce a per-layer
    # quantization scheme where no quantized tensors escape their layer.
    for layer_name, qp in quant_params_struct.items():
        print(f"Applying per-layer quants: {layer_name}")
        layer_theta = theta(layer_name)
        apply_per_layer_quant(layer_theta, qp)
        break

    # TODO: Post-process to introduce fused cross-layer connections.
    return

    dataset.save(args.output_irpa_file, io_report_callback=print)


if __name__ == "__main__":
    main()
