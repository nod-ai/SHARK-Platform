# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import pytest
from pytest import FixtureRequest
from typing import Optional, Any
from pytest_html import extras, hooks


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

    # TODO: Remove all hardcoded paths in CI tests
    parser.addoption(
        "--llama3-8b-tokenizer-path",
        type=Path,
        action="store",
        help="Llama3.1 8b tokenizer path, defaults to 30F CI system path",
    )

    parser.addoption(
        "--llama3-8b-f16-model-path",
        type=Path,
        action="store",
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
        help="Llama3.1 405b tokenizer path, defaults to 30F CI system path",
    )

    parser.addoption(
        "--llama3-405b-f16-model-path",
        type=Path,
        action="store",
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
        default="sharktank/tests/evaluate/baseline_perplexity_scores.json",
        help="Llama3.1 8B & 405B model baseline perplexity scores",
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
        help="Specify the iree-hip target version (e.g., gfx942)",
    )

    parser.addoption(
        "--iree-hal-target-backends",
        action="store",
        default="rocm",
        help="Specify the iree-hal target backend (e.g., rocm)",
    )

    parser.addoption(
        "--tensor-parallelism-size",
        action="store",
        type=int,
        default=1,
        help="Number of devices for tensor parallel sharding",
    )

    parser.addoption(
        "--bs",
        action="store",
        type=int,
        default=4,
        help="Batch size for mlir export",
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
def tensor_parallelism_size(request: FixtureRequest) -> Optional[str]:
    return set_fixture_from_cli_option(
        request, "tensor_parallelism_size", "tensor_parallelism_size"
    )


@pytest.fixture(scope="class")
def baseline_perplexity_scores(request: FixtureRequest) -> Optional[str]:
    return set_fixture_from_cli_option(
        request, "baseline_perplexity_scores", "baseline_perplexity_scores"
    )


@pytest.fixture(scope="class")
def batch_size(request: FixtureRequest) -> Optional[str]:
    return set_fixture_from_cli_option(request, "bs", "batch_size")


@pytest.fixture(scope="class")
def get_model_artifacts(request: FixtureRequest):
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
    return model_path


@pytest.fixture(scope="class")
def get_iree_flags(request: FixtureRequest):
    model_path = {}
    model_path["iree_device"] = set_fixture_from_cli_option(
        request, "--iree-device", "iree_device"
    )
    model_path["iree_hip_target"] = set_fixture_from_cli_option(
        request, "--iree-hip-target", "iree_hip_target"
    )
    model_path["iree_hal_target_backends"] = set_fixture_from_cli_option(
        request, "--iree-hal-target-backends", "iree_hal_target_backends"
    )

# Hook to add extra columns or modify the table row in the pytest-html report
def pytest_html_results_table_header(cells):
    cells.insert(2, hooks.html.Cell("XFail Reason"))

def pytest_html_results_table_row(report, cells):
    if hasattr(report, "wasxfail"):
        cells.insert(2, hooks.html.Cell(report.wasxfail))
    else:
        cells.insert(2, hooks.html.Cell(""))

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and hasattr(report, "wasxfail"):
        report.wasxfail = getattr(report, "wasxfail", "")
