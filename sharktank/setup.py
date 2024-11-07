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
VERSION_INFO_RC_FILE = os.path.join(SETUPPY_DIR, "version_info_rc.json")


def load_version_info(version_file):
    with open(version_file, "rt") as f:
        return json.load(f)


try:
    version_info = load_version_info(VERSION_INFO_RC_FILE)
except FileNotFoundError:
    print("version_info_rc.json not found. Default to dev build")
    version_info = load_version_info(VERSION_INFO_FILE)

PACKAGE_VERSION = version_info.get("package-version")
print(f"Using PACKAGE_VERSION: '{PACKAGE_VERSION}'")

setup(
    version=f"{PACKAGE_VERSION}",
)
