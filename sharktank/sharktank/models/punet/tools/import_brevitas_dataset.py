# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Imports Brevitas pre-processed weights and quantization config into a 
Dataset.

Usage:
  python -m sharktank.models.punet.import_hf_dataset \
    --output-irpa-file ~/models/punet/punet_fp16.irpa \
    --config-json ~/models/stable-diffusion-xl-base-1.0/unet/config.json

The resulting dataset has all tensors as nested in the original model.
Properties are separated into a "meta" dict (for "_" prefixed props) and an
"hparams" dict.

Default flag values assume that there is a quant_param.json and 
params.safetensors adjacent to the HF config.json file.
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


def apply_per_layer_quant(
    ds: Dataset, layer_name: str, qp, updated_tensors: dict[str, InferenceTensor]
):
    layer_theta = ds.root_theta(layer_name)

    # The config file has tensors as 1d and a _shape suffixed array with the
    # concrete shape.
    def _get_json_tensor(name: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
        data_1d = qp.get(name)
        if data_1d is None:
            return None
        shape = qp[f"{name}_shape"]
        return torch.tensor(data_1d, dtype=dtype).reshape(shape)

    input_scale = _get_json_tensor("input_scale", torch.float32)
    input_zp = _get_json_tensor("input_zp", torch.uint8)
    weight_scale = _get_json_tensor("weight_scale", torch.float32)
    weight_zp = _get_json_tensor("weight_zp", torch.uint8)
    assert (
        weight_scale is not None and weight_zp is not None
    ), f"Could not find weight scale (in {qp.keys()})"
    assert input_scale is not None, f"Could not find input scale (in {qp.keys()})"

    # Weight scaling.
    weight = layer_theta.tensor("weight")
    weight_dtype = weight.as_torch().dtype
    # TODO: There is ambiguity with respect to the axis of quantization and the
    # shape of the scale. I think the intent was that the scale should have a
    # broadcast ready shape, but they are all 1d. This model only does axis-0
    # quantization, so we just assert that matches for now.
    assert weight.shape[0] == weight_scale.shape[0], "Mismatched quantization axis"

    # There is an implicit assumption that the weight is asym (uint8) quantized.
    # Our quantizer uses scale/offset nomenclature. The offset maps to
    # zero-point, and the scale maps to the *dequant* scale (so terms differ
    # by reciprocal).
    weight_quantizer = StaticScaledQuantizer(
        scale=1.0 / weight_scale,
        reciprocal_scale=weight_scale,
        axis=0,
        offset=None if torch.count_nonzero(weight_zp) == 0 else weight_zp,
        dtype=torch.uint8,
    )
    weight_quant = weight_quantizer.quantize(weight, name=weight.name)
    updated_tensors[weight_quant.name] = weight_quant
    # Spot check that things look sane.
    # weight_dequant = weight_quant.unpack().dequant()
    # print(f"ORIG:\n{weight.as_torch()[0]}")
    # print(f"DEQUANT:\n{weight_dequant[0]}")

    # Input scaling.
    # Assume per tensor scaling of input.
    assert torch.count_nonzero(input_zp) == 0
    assert len(input_scale.shape) == 0
    input_quantizer = StaticScaledQuantizer(
        name=f"{layer_name}.q_input",
        scale=1.0 / input_scale,
        reciprocal_scale=input_scale,
        dtype=torch.int8,
    )
    updated_tensors[input_quantizer.name] = input_quantizer

    # Output dequant back to high precision.
    output_scale = input_quantizer.scale * weight_quantizer.scale
    output_quantizer = StaticScaledQuantizer(
        name=f"{layer_name}.dq_output",
        scale=output_scale,
        axis=0,
        dtype=weight_dtype,
    )
    updated_tensors[output_quantizer.name] = output_quantizer

    # Optional activation pre-multiplier.
    smoothquant_mul = _get_json_tensor("smoothquant_mul", dtype=weight_dtype)
    if smoothquant_mul is not None:
        premul_input = DefaultPrimitiveTensor(
            name=f"{layer_name}.premul_input",
            data=smoothquant_mul,
        )
        updated_tensors[premul_input.name] = premul_input


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
    updated_tensors: dict[str, InferenceTensor] = {}
    for layer_name, qp in quant_params_struct.items():
        print(f"Applying per-layer quants: {layer_name}")
        apply_per_layer_quant(ds, layer_name, qp, updated_tensors)

    # Apply updates into a new Theta.
    flat_tensors = theta.flatten()
    flat_tensors.update(updated_tensors)
    ds.root_theta = Theta(flat_tensors)

    # TODO: Post-process to introduce fused cross-layer connections.

    ds.save(args.output_irpa_file, io_report_callback=print)


if __name__ == "__main__":
    main()
