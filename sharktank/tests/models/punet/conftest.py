# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from pathlib import Path
from typing import Optional


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
        "--caching",
        action="store_true",
        default=False,
        help="Load cached results if present instead of recomputing.",
    )


@pytest.fixture(scope="session")
def mlir_path(pytestconfig: pytest.Config) -> Optional[Path]:
    return pytestconfig.getoption("mlir")


@pytest.fixture(scope="session")
def module_path(pytestconfig: pytest.Config) -> Optional[Path]:
    return pytestconfig.getoption("module")


@pytest.fixture(scope="session")
def parameters_path(pytestconfig: pytest.Config) -> Optional[Path]:
    return pytestconfig.getoption("parameters")


@pytest.fixture(scope="session")
def caching(pytestconfig: pytest.Config) -> Optional[Path]:
    return pytestconfig.getoption("caching")
