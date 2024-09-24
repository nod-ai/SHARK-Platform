# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Utilities for managing dependencies.

The overall shortfin namespace contains components with optional dependencies.
This module provides support for reacting to dependcy issues.
"""


class ShortfinDepNotFoundError(Exception):
    """Raised from a ModuleNotFoundError for a missing or incorrect dep."""

    def __init__(
        self, caller_name: str, package_name: str, extras_name: str | None = None
    ):
        super().__init__()
        self.caller_name = caller_name.removesuffix("._deps")
        self.package_name = package_name
        self.extras_name = extras_name

    def __str__(self):
        msg = (
            f"Shortfin is missing a dependency to use {self.caller_name}. "
            f"This is typically available via `pip install {self.package_name}`"
        )
        if self.extras_name:
            msg += (
                f" (or by installing with an extra like "
                f"`pip install shortfin[{self.extras_name}])"
            )
        return msg


ShortfinDepNotFoundError.__name__ = "ShortfinDepNotFoundError"
