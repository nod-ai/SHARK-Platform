# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from os import PathLike

from ...export import export_static_model_mlir
from ...tools.import_hf_dataset import import_hf_dataset
from .flux import FluxModelV1, FluxParams
from ...types import Dataset
from ...utils.hf_datasets import get_dataset

flux_transformer_default_batch_sizes = [4]


def export_flux_transformer_model_mlir(
    model: FluxModelV1,
    output_path: PathLike,
    batch_sizes: list[int] = flux_transformer_default_batch_sizes,
):
    export_static_model_mlir(model, output_path=output_path, batch_sizes=batch_sizes)


def export_flux_transformer_from_hugging_face(
    repo_id: str,
    mlir_output_path: PathLike,
    parameters_output_path: PathLike,
    batch_sizes: list[int] = flux_transformer_default_batch_sizes,
):
    hf_dataset = get_dataset(
        repo_id,
    ).download()

    import_hf_dataset(
        config_json_path=hf_dataset["config"][0],
        param_paths=hf_dataset["parameters"],
        output_irpa_file=parameters_output_path,
    )

    dataset = Dataset.load(parameters_output_path)
    model = FluxModelV1(
        theta=dataset.root_theta,
        params=FluxParams.from_hugging_face_properties(dataset.properties),
    )
    export_flux_transformer_model_mlir(
        model, output_path=mlir_output_path, batch_sizes=batch_sizes
    )
