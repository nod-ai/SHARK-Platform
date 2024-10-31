# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
from pathlib import Path

from setuptools import setup

SETUPPY_DIR = os.path.realpath(os.path.dirname(__file__))

# Setup and get version information.
VERSION_INFO_FILE = os.path.join(SETUPPY_DIR, "version_info.json")


def load_version_info():
    with open(VERSION_INFO_FILE, "rt") as f:
        return json.load(f)


version_info = load_version_info()
PACKAGE_VERSION = version_info["package-version"]

setup(
    version=f"{PACKAGE_VERSION}",
)
