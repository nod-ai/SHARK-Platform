# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# The proper way to import this package is via:
#   from _shortfin import lib as sfl

from typing import TYPE_CHECKING

import os
import sys
import warnings

if TYPE_CHECKING:
    from _shortfin_default import lib
else:
    variant = os.getenv("SHORTFIN_PY_RUNTIME", "default")

    if variant == "tracy":
        try:
            from _shortfin_tracy import lib
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Shortfin Tracy runtime requested via SHORTFIN_PY_RUNTIME but it is not enabled in this build"
            )
        print("-- Using Tracy runtime (SHORTFIN_PY_RUNTIME=tracy)", file=sys.stderr)
    else:
        if variant != "default":
            warnings.warn(
                f"Unknown value for SHORTFIN_PY_RUNTIME env var ({variant}): Using default"
            )
        from _shortfin_default import lib
