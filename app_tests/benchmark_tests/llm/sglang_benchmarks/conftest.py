# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import pytest
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
from integration_tests.llm.utils import (
    compile_model,
    export_paged_llm_v1,
    download_with_hf_datasets,
)


@pytest.fixture(scope="module")
def pre_process_model(request, tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("sglang_benchmark_test")

    model_name = request.param["model_name"]
    model_param_file_name = request.param["model_param_file_name"]
    settings = request.param["settings"]
    batch_sizes = request.param["batch_sizes"]

    mlir_path = tmp_dir / "model.mlir"
    config_path = tmp_dir / "config.json"
    vmfb_path = tmp_dir / "model.vmfb"

    model_path = tmp_dir / model_param_file_name
    download_with_hf_datasets(tmp_dir, model_name)

    export_paged_llm_v1(mlir_path, config_path, model_path, batch_sizes)

    compile_model(mlir_path, vmfb_path, settings)

    return tmp_dir
