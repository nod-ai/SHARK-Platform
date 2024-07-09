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
import sys
import torch

from ....types import *

# It is possible to import quant params from stock unet weights for testing.
# Quality won't be great but needs SMOOTHQUANT prescaling disabled to work
# at all.
IMPORT_SMOOTHQUANT_PRESCALE = True

# Quantizing the bias can produce better fusions but puts more pressure on
# datatype ranges.
QUANTIZE_BIAS = True


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
    root_theta: Theta, layer_name: str, qp, updated_tensors: dict[str, InferenceTensor]
):
    layer_theta = root_theta(layer_name)
    weight = layer_theta.tensor("weight")
    weight_dtype = weight.as_torch().dtype

    # The config file has tensors as 1d and a _shape suffixed array with the
    # concrete shape.
    def _get_json_tensor(name: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
        data_1d = qp.get(name)
        if data_1d is None:
            return None
        shape = qp[f"{name}_shape"]
        return torch.tensor(data_1d, dtype=dtype).reshape(shape)

    # Optional activation pre-multiplier.
    if IMPORT_SMOOTHQUANT_PRESCALE:
        smoothquant_mul = _get_json_tensor("smoothquant_mul", dtype=weight_dtype)
        if smoothquant_mul is not None:
            premul_input = DefaultPrimitiveTensor(
                name=f"{layer_name}.premul_input",
                data=smoothquant_mul,
            )
            updated_tensors[premul_input.name] = premul_input

    input_scale = _get_json_tensor("input_scale", torch.float32)
    weight_scale = _get_json_tensor("weight_scale", torch.float32)
    # In the JSON file, zero points are purely positive numbers representing
    # the zero point assuming a uint8 datatype. Since we prefer to use
    # signed arithmetic, since that is better accelerated, we offset these by
    # -128. By decoding them as uint8, it validates our range assumption.
    # Then we widen/cast to offset.
    weight_zp = _get_json_tensor("weight_zp", torch.uint8)
    if weight_zp is not None:
        weight_zp = (weight_zp.to(dtype=torch.int32) - 128).to(torch.int8)

    # In the current version, we assume that the input is not offset and is
    # per-tensor quantized for signed arithmetic.
    input_zp = _get_json_tensor("input_zp", torch.int8)
    if input_zp is not None:
        assert torch.count_nonzero(input_zp) == 0

    if (
        input_scale is None
        and input_zp is None
        and weight_scale is None
        and weight_zp is None
    ):
        # Non quantized layer that has had possible adjustments above.
        return

    # Quantized layer must have all quantization info.
    assert (
        weight_scale is not None and weight_zp is not None
    ), f"Could not find weight scale (in {qp.keys()}) for {layer_name}"
    assert (
        input_scale is not None
    ), f"Could not find input scale (in {qp.keys()}) for {layer_name}"

    # Weight scaling.
    # There is an implicit assumption that the weight is asym (uint8) quantized.
    # Our quantizer uses scale/offset nomenclature. The offset maps to
    # zero-point, and the scale maps to the *dequant* scale (so terms differ
    # by reciprocal).
    weight_quantizer = StaticScaledQuantizer(
        scale=1.0 / weight_scale,
        reciprocal_scale=weight_scale,
        offset=None if torch.count_nonzero(weight_zp) == 0 else weight_zp,
        dtype=torch.int8,
    )
    weight_quant = weight_quantizer.quantize(weight, name=weight.name)
    updated_tensors[weight_quant.name] = weight_quant
    # Spot check that things look sane.
    weight_dequant = weight_quant.unpack().dequant()
    weight_diff = weight.as_torch() - weight_dequant

    # Bias/output scaling.
    bias = layer_theta.optional_tensor("bias")
    if QUANTIZE_BIAS and bias is not None:
        # If the bias is present, it dictates the overall output quantization
        # and will not be checked for correct parameters at runtime. It must
        # be quantized to match properly.
        bias_scale = 1.0 / (input_scale * weight_scale)
        bias_quantizer = StaticScaledQuantizer(
            scale=bias_scale,
            dtype=torch.int32,
            disable_saturate=True,
        )
        bias_quant = bias_quantizer.quantize(bias, name=bias.name)
        updated_tensors[bias_quant.name] = bias_quant
        # Spot check that things look sane.
        bias_dequant = bias_quant.unpack().dequant()
        bias_diff = bias.as_torch() - bias_dequant

    # Input scaling.
    # Assume per tensor scaling of input.
    assert len(input_scale.shape) == 0
    input_quantizer = StaticScaledQuantizer(
        name=f"{layer_name}.q_input",
        scale=1.0 / input_scale,
        reciprocal_scale=input_scale,
        dtype=torch.int8,
    )
    updated_tensors[input_quantizer.name] = input_quantizer


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
        default=Path("params.safetensors"),
        help="Parameter file name, relative to config.json",
    )
    parser.add_argument(
        "--quant-params",
        type=Path,
        default=Path("quant_param.json"),
        help="Quantization parameters",
    )
    parser.add_argument(
        "--base-params",
        type=Path,
        help="Base parameters to initialize from (will be augmented with quantized)",
    )
    args = cli.parse(parser, args=argv)

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
        quant_theta = _load_theta(st)
    base_theta = None
    if args.base_params is not None:
        print("Initializing from base parameters:", args.base_params)
        with safetensors.safe_open(
            args.base_params, framework="pt", device="cpu"
        ) as st:
            base_theta = _load_theta(st)

    ds = Dataset(dataset_props, quant_theta if base_theta is None else base_theta)

    # The quant_params_struct has quantization parameter structs keyed by full
    # layer name. We process each of these in turn to produce a per-layer
    # quantization scheme where no quantized tensors escape their layer.
    updated_tensors: dict[str, InferenceTensor] = {}
    for layer_name, qp in quant_params_struct.items():
        print(f"Applying per-layer quants: {layer_name}")
        apply_per_layer_quant(quant_theta, layer_name, qp, updated_tensors)

    # Apply updates into a new Theta.
    theta = base_theta if base_theta is not None else quant_theta
    flat_tensors = theta.flatten()
    flat_tensors.update(updated_tensors)
    ds.root_theta = Theta(flat_tensors)

    # TODO: Post-process to introduce fused cross-layer connections.

    ds.save(args.output_irpa_file, io_report_callback=print)


if __name__ == "__main__":
    main(sys.argv[1:])
