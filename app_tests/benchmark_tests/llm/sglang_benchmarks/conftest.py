# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import pytest
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from integration_tests.llm.utils import compile_model, export_paged_llm_v1, download_with_hf_datasets


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

    config = {
        "module_name": "module",
        "module_abi_version": 1,
        "max_seq_len": 131072,
        "attn_head_count": 8,
        "attn_head_dim": 128,
        "prefill_batch_sizes": batch_sizes,
        "decode_batch_sizes": batch_sizes,
        "transformer_block_count": 32,
        "paged_kv_cache": {"block_seq_stride": 16, "device_block_count": 256},
    }
    with open(config_path, "w") as file:
        json.dump(config, file)

    compile_model(mlir_path, vmfb_path, settings)

    return tmp_dir


