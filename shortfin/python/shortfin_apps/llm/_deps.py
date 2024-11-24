# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from shortfin.support.deps import ShortfinDepNotFoundError
import sys

shortfin_llm_deps = [
    "tokenizers",
    "dataclasses_json",
]

for dep in shortfin_llm_deps:
    try:
        __import__(dep)
    except ModuleNotFoundError as e:
        if "pytest" in sys.modules:
            import pytest

            pytest.skip(f"Shortfin LLM dependency not available: {dep}")
        else:
            raise ShortfinDepNotFoundError(__name__, dep) from e
