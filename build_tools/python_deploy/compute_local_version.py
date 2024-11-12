#!/usr/bin/env python3
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This scripts grabs the X.Y.Z[.dev]` version identifier from a
# `version.json` and writes the corresponding
# `X.Y.ZrcYYYYMMDD` version identifier to `version_local.json`.

import argparse
from pathlib import Path
import json
from datetime import datetime

from packaging.version import Version


parser = argparse.ArgumentParser()
parser.add_argument("path", type=Path)
args = parser.parse_args()

VERSION_FILE = args.path / "version.json"
VERSION_FILE_LOCAL = args.path / "version_local.json"


def load_version_info():
    with open(VERSION_FILE, "rt") as f:
        return json.load(f)


def write_version_info():
    with open(VERSION_FILE_LOCAL, "w") as f:
        json.dump(version_local, f, indent=2)
        f.write("\n")


version_info = load_version_info()

PACKAGE_VERSION = version_info.get("package-version")
PACKAGE_BASE_VERSION = Version(PACKAGE_VERSION).base_version
PACKAGE_LOCAL_VERSION = (
    PACKAGE_BASE_VERSION + "rc" + datetime.today().strftime("%Y%m%d")
)

version_local = {"package-version": PACKAGE_LOCAL_VERSION}

write_version_info()

print(PACKAGE_LOCAL_VERSION)
