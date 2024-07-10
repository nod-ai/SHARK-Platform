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

    layer_theta = root_theta(layer_name)
    weight = layer_theta.tensor("weight").as_torch()
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

    layer_parent = ".".join(layer_name.split('.')[:-1])
    if "qkv" in layer_name:
        print("qkv layer found")
        print("weight_quant_scale shape: ", weight_quant_scale.shape)
        print("layer_parent: ", layer_parent)
        torch_weight = weight
        q_weight = torch_weight[:8192, :]
        k_weight = torch_weight[8192:9216, :]
        v_weight = torch_weight[9216:, :]

    if "qkv" in layer_name:
        q_weight_quant = PlanarQuantizedTensor(
                shape=q_weight.shape,
                name=layer_parent + '.q_proj.weight',
                layout=TensorScaledLayout(
                    shape=q_weight.shape,
                    d=1.0/weight_quant_scale,
                    qs=q_weight.to(dtype=torch.float8_e4m3fnuz),
                    m=weight_quant_zero_point,
                    dtype=torch.float16,  # Original dtype.
                )
        )
        k_weight_quant = PlanarQuantizedTensor(
                shape=k_weight.shape,
                name=layer_parent + '.k_proj.weight',
                layout=TensorScaledLayout(
                    shape=k_weight.shape,
                    d=1.0/weight_quant_scale,
                    qs=k_weight.to(dtype=torch.float8_e4m3fnuz),
                    m=weight_quant_zero_point,
                    dtype=torch.float16,  # Original dtype.
                )
        )
        v_weight_quant = PlanarQuantizedTensor(
                shape=v_weight.shape,
                name=layer_parent + '.v_proj.weight',
                layout=TensorScaledLayout(
                    shape=v_weight.shape,
                    d=1.0/weight_quant_scale,
                    qs=v_weight.to(dtype=torch.float8_e4m3fnuz),
                    m=weight_quant_zero_point,
                    dtype=torch.float16,  # Original dtype.
                )
        )
        print(q_weight_quant.name)
        print(k_weight_quant.name)
        print(v_weight_quant.name)
        updated_tensors[q_weight_quant.name] = q_weight_quant
        updated_tensors[k_weight_quant.name] = k_weight_quant
        updated_tensors[v_weight_quant.name] = v_weight_quant
        #updated_tensors[layer_name] = None
    else:
        weight_quant = PlanarQuantizedTensor(
            shape=weight.shape,
            name=layer_name + '.weight',
            layout=TensorScaledLayout(
                shape=weight.shape,
                d=1.0/weight_quant_scale,
                qs=weight.to(dtype=torch.float8_e4m3fnuz),
                m=weight_quant_zero_point,
                dtype=torch.float16,  # Original dtype.
            )
        )
        print(weight_quant.name)
        updated_tensors[weight_quant.name] = weight_quant
        # Spot check that things look sane.
        #weight_dequant = weight_quant.unpack().dequant()
        #torch.testing.assert_close(weight, weight_dequant, atol=3, rtol=3)


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
    print("hi")
