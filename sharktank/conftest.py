# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import pytest
from pytest import FixtureRequest
from typing import Optional, Any


# Tests under each top-level directory will get a mark.
TLD_MARKS = {
    "tests": "unit",
    "integration": "integration",
}


def pytest_collection_modifyitems(items, config):
    # Add marks to all tests based on their top-level directory component.
    root_path = Path(__file__).resolve().parent
    for item in items:
        item_path = Path(item.path)
        rel_path = item_path.relative_to(root_path)
        tld = rel_path.parts[0]
        mark = TLD_MARKS.get(tld)
        if mark:
            item.add_marker(mark)


def pytest_addoption(parser):
    parser.addoption(
        "--mlir",
        type=Path,
        default=None,
        help="Path to exported MLIR program. If not specified a temporary file will be used.",
    )
    parser.addoption(
        "--module",
        type=Path,
        default=None,
        help="Path to exported IREE module. If not specified a temporary file will be used.",
    )
    parser.addoption(
        "--parameters",
        type=Path,
        default=None,
        help="Exported model parameters. If not specified a temporary file will be used.",
    )
    parser.addoption(
        "--prefix",
        type=str,
        default=None,
        help=(
            "Path prefix for test artifacts. "
            "Other arguments may override this for specific values."
        ),
    )
    parser.addoption(
        "--caching",
        action="store_true",
        default=False,
        help="Load cached results if present instead of recomputing.",
    )

    parser.addoption(
        "--longrun",
        action="store_true",
        dest="longrun",
        default=False,
        help="Enable long and slow tests",
    )

    parser.addoption(
        "--llama3-8b-tokenizer-path",
        type=Path,
        action="store",
        default="/data/extra/models/llama3.1_8B/tokenizer_config.json",
        help="Llama3.1 8b tokenizer path, defaults to 30F CI system path",
    )

    parser.addoption(
        "--llama3-8b-f16-model-path",
        type=Path,
        action="store",
        default="/data/extra/models/llama3.1_8B/llama8b_f16.irpa",
        help="Llama3.1 8b model path, defaults to 30F CI system path",
    )

    parser.addoption(
        "--llama3-8b-fp8-model-path",
        type=Path,
        action="store",
        default=None,
        help="Llama3.1 8b fp8 model path",
    )

    parser.addoption(
        "--llama3-405b-tokenizer-path",
        type=Path,
        action="store",
        default="/data/extra/models/llama3.1_405B/tokenizer_config.json",
        help="Llama3.1 405b tokenizer path, defaults to 30F CI system path",
    )

    parser.addoption(
        "--llama3-405b-f16-model-path",
        type=Path,
        action="store",
        default="/data/extra/models/llama3.1_405B/llama405b_fp16.irpa",
        help="Llama3.1 405b model path, defaults to 30F CI system path",
    )

    parser.addoption(
        "--llama3-405b-fp8-model-path",
        type=Path,
        action="store",
        default=None,
        help="Llama3.1 405b fp8 model path",
    )

    parser.addoption(
        "--baseline-perplexity-scores",
        type=Path,
        action="store",
        default="/home/aramalin/SHARK-Platform/sharktank/tests/evaluate/baseline_perplexity_scores.json",
        help="Llama3.1 8B & 405B model baseline perplexity scores",
    )

    parser.addoption(
        "--llama3-8b-f16-vmfb-path",
        type=Path,
        action="store",
        default="/data/extra/models/llama3.1_8B/llama8b_f16.vmfb",
        help="Llama3.1 8b fp16 vmfb path, defaults to 30F CI system path",
    )

    parser.addoption(
        "--llama3-405b-f16-vmfb-path",
        type=Path,
        action="store",
        default="/data/extra/models/llama3.1_405B/llama405b_fp16.vmfb",
        help="Llama3.1 405b fp16 vmfb path, defaults to 30F CI system path",
    )

    parser.addoption(
        "--iree-device",
        type=str,
        action="store",
        help="List an IREE device from iree-run-module --list_devices",
    )

    parser.addoption(
        "--iree-hip-target",
        action="store",
        default="gfx942",
        help="Specify the iree-hip target version (e.g., gfx942)",
    )


def set_fixture_from_cli_option(
    request: FixtureRequest,
    cli_option_name: str,
    class_attribute_name: Optional[str] = None,
) -> Optional[Any]:
    res = request.config.getoption(cli_option_name)
    if request.cls is None:
        return res
    else:
        if class_attribute_name is None:
            class_attribute_name = cli_option_name
        setattr(request.cls, class_attribute_name, res)


@pytest.fixture(scope="class")
def mlir_path(request: FixtureRequest) -> Optional[Path]:
    return set_fixture_from_cli_option(request, "mlir", "mlir_path")


@pytest.fixture(scope="class")
def module_path(request: FixtureRequest) -> Optional[Path]:
    return set_fixture_from_cli_option(request, "module", "module_path")


@pytest.fixture(scope="class")
def parameters_path(request: FixtureRequest) -> Optional[Path]:
    return set_fixture_from_cli_option(request, "parameters", "parameters_path")


@pytest.fixture(scope="class")
def path_prefix(request: FixtureRequest) -> Optional[str]:
    return set_fixture_from_cli_option(request, "prefix", "path_prefix")


@pytest.fixture(scope="class")
def caching(request: FixtureRequest) -> Optional[bool]:
    return set_fixture_from_cli_option(request, "caching")


@pytest.fixture(scope="class")
def iree_hip_target_type(request: FixtureRequest) -> Optional[str]:
    return set_fixture_from_cli_option(
        request, "iree_hip_target", "iree_hip_target_type"
    )


@pytest.fixture(scope="class")
def get_model_path(request: FixtureRequest):
    model_path = {}
    model_path["llama3_8b_tokenizer_path"] = set_fixture_from_cli_option(
        request, "--llama3-8b-tokenizer-path", "llama3_8b_tokenizer"
    )
    model_path["llama3_8b_f16_model_path"] = set_fixture_from_cli_option(
        request, "--llama3-8b-f16-model-path", "llama3_8b_f16_model"
    )
    model_path["llama3_8b_fp8_model_path"] = set_fixture_from_cli_option(
        request, "--llama3-8b-fp8-model-path", "llama3_8b_fp8_model"
    )
    model_path["llama3_405b_tokenizer_path"] = set_fixture_from_cli_option(
        request, "--llama3-405b-tokenizer-path", "llama3_405b_tokenizer"
    )
    model_path["llama3_405b_f16_model_path"] = set_fixture_from_cli_option(
        request, "--llama3-405b-f16-model-path", "llama3_405b_f16_model"
    )
    model_path["llama3_405b_fp8_model_path"] = set_fixture_from_cli_option(
        request, "--llama3-405b-fp8-model-path", "llama3_405b_fp8_model"
    )
    model_path["baseline_perplexity_scores"] = set_fixture_from_cli_option(
        request, "--baseline-perplexity-scores", "baseline_perplexity_scores"
    )
    model_path["llama3_8b_f16_vmfb"] = set_fixture_from_cli_option(
        request, "--llama3-8b-f16-vmfb-path", "llama3_8b_f16_vmfb"
    )
    model_path["llama3_405b_f16_vmfb"] = set_fixture_from_cli_option(
        request, "--llama3-405b-f16-vmfb-path", "llama3_405b_f16_vmfb"
    )
    model_path["iree_device"] = set_fixture_from_cli_option(
        request, "--iree-device", "iree_device"
    )
    return model_path
