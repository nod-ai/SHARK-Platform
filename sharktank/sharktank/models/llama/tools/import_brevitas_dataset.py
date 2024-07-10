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

from sharktank.types import *

# It is possible to import quant params from stock unet weights for testing.
# Quality won't be great but needs SMOOTHQUANT prescaling disabled to work
# at all.
IMPORT_SMOOTHQUANT_PRESCALE = False

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

def as_torch_or_none(tensor: Optional[InferenceTensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.as_torch()

def apply_per_layer_quant(
    root_theta: Theta, layer_name: str, updated_tensors: dict[str, InferenceTensor]
):
    if "kv_cache_scaling_factor" in layer_name:
        return
    layer_theta = root_theta(layer_name)
    weight = layer_theta.tensor("weight")
    weight_dtype = weight.as_torch().dtype
    weight_quant_scale = layer_theta.tensor("weight_quant_scale").as_torch()
    weight_quant_zero_point = layer_theta.optional_tensor("weight_quant_zero_point")
    if weight_quant_zero_point == None:
        weight_quant_zero_point = torch.zeros(1, dtype=torch.float32)
    else:
        weight_quant_zero_point = weight_quant_zero_point.as_torch()
    input_quant_scale = as_torch_or_none(layer_theta.optional_tensor("input_quant_scale"))

    if weight_quant_scale is None:
        print("weight quant scale not found for layer ", layer_name)
        return

    # Weight scaling.
    # There is an implicit assumption that the weight is asym (uint8) quantized.
    # Our quantizer uses scale/offset nomenclature. The offset maps to
    # zero-point, and the scale maps to the *dequant* scale (so terms differ
    # by reciprocal).
    weight_quantizer = StaticScaledQuantizer(
        reciprocal_scale=1.0 / weight_quant_scale,
        scale=weight_quant_scale,
        offset=None if torch.count_nonzero(weight_quant_zero_point) == 0 else weight_quant_zero_point,
        dtype=torch.float8_e4m3fn,
    )
    weight_quant = weight_quantizer.quantize(weight, name=weight.name)
    updated_tensors[weight_quant.name] = weight_quant
    # Spot check that things look sane.
    weight_dequant = weight_quant.unpack().dequant()
    torch.testing.assert_close(weight.as_torch(), weight_dequant, atol=3, rtol=3)
    # Bias/output scaling.
    bias = layer_theta.optional_tensor("bias")
    if QUANTIZE_BIAS and bias is not None:
        # If the bias is present, it dictates the overall output quantization
        # and will not be checked for correct parameters at runtime. It must
        # be quantized to match properly.
        bias_scale = 1.0 / (input_quant_scale * weight_quant_scale)
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
    #assert len(input_quant_scale.shape) == 0
    #input_quantizer = StaticScaledQuantizer(
    #    name=f"{layer_name}.q_input",
    #    reciprocal_scale=1.0 / input_quant_scale,
    #    scale=input_quant_scale,
    #    dtype=torch.float8_e4m3fn,
    #)
    #updated_tensors[input_quantizer.name] = input_quantizer


def main(argv):
    from sharktank.utils import cli

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
        "--base-params",
        type=Path,
        help="Base parameters to initialize from (will be augmented with quantized)",
    )
    args = cli.parse(parser, args=argv)

    config_json_path: Path = args.config_json
    params_path: Path = args.params
    # Construct the pre-transform dataset.
    dataset_props = _get_dataset_props(_load_json(config_json_path))
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
    model_layers = ["model.layers."+str(i) for i in range(80)]
    sub_layers = ['mlp.down_proj', 'mlp.up_proj', 'self_attn.o_proj', 'self_attn.qkv' ]
    for layer in model_layers:
        for sub in sub_layers:

            layer_name = layer + '.' + sub
            #if layern_name not in ["quantization", "decoder_type", 
            print(f"Applying per-layer quants: {layer_name}")
            apply_per_layer_quant(quant_theta, layer_name, updated_tensors)

    # Apply updates into a new Theta.
    theta = base_theta if base_theta is not None else quant_theta
    flat_tensors = theta.flatten()
    flat_tensors.update(updated_tensors)
    ds.root_theta = Theta(flat_tensors)

    # TODO: Post-process to introduce fused cross-layer connections.

    ds.save(args.output_irpa_file, io_report_callback=print)


if __name__ == "__main__":
    main(sys.argv[1:])
