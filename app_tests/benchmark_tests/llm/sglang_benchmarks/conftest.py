# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import json
import logging
import os
import pytest
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
from integration_tests.llm.utils import (
    compile_model,
    end_log_group,
    export_paged_llm_v1,
    download_with_hf_datasets,
    start_log_group,
)

logger = logging.getLogger(__name__)

MODEL_DIR_CACHE = {}


@pytest.fixture(scope="module")
def pre_process_model(request, tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("sglang_benchmark_test")

    logger.info(
        "Preparing model artifacts..." + start_log_group("Preparing model artifacts")
    )

    param_key = hashlib.md5(str(request.param).encode()).hexdigest()
    if (directory := MODEL_DIR_CACHE.get(param_key)) is not None:
        logger.info(
            f"Reusing existing model artifacts directory: {directory}" + end_log_group()
        )
        return MODEL_DIR_CACHE[param_key]

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

    logger.info("Model artifacts setup successfully" + end_log_group())
    MODEL_DIR_CACHE[param_key] = tmp_dir
    return tmp_dir


@pytest.fixture(scope="module")
def write_config(request, pre_process_model):
    batch_sizes = request.param["batch_sizes"]
    prefix_sharing_algorithm = request.param["prefix_sharing_algorithm"]

    logger.info("Writing config file..." + start_log_group("Writing config file"))

    config_path = (
        pre_process_model
        / f"{'_'.join(str(bs) for bs in batch_sizes)}_{prefix_sharing_algorithm}.json"
    )

    yield config_path


def pytest_addoption(parser):
    parser.addoption(
        "--port",
        action="store",
        default="30000",
        help="Port that SGLang server is running on",
    )


@pytest.fixture(scope="module")
def sglang_args(request):
    return request.config.getoption("--port")
