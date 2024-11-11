#!/usr/bin/env python3
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This scripts writes the `packaging/shark-platform/requirements.txt` file
# and pins the versions of the dependencies accordingly. For nighly releases,
#  * sharktank
#  * shortfin
# get pinned to the corresponding nighly version. For stable releases,
# * iree-base-compiler
# * iree-base-runtime
# * iree-turbine
# * sharktank
# * shortfin
# get pinned to the corresponding `X.Y.*` version.


from pathlib import Path
import json

from packaging.version import Version


THIS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = THIS_DIR.parent.parent

VERSION_FILE_LOCAL = REPO_ROOT / "packaging/shark-platform/version_local.json"
REQUIREMENTS_TXT = REPO_ROOT / "packaging/shark-platform/requirements.txt"


def load_version_info(version_file):
    with open(version_file, "rt") as f:
        return json.load(f)


def write_requirements(package_list, package_version):
    with open(REQUIREMENTS_TXT, "w") as f:
        for package in package_list:
            PINNED_PACKAGE = package + "==" + package_version
            f.write("%s\n" % PINNED_PACKAGE)


metapackage_version = load_version_info(VERSION_FILE_LOCAL)
PACKAGE_VERSION = metapackage_version.get("package-version")

nightly_packages_list = ["sharktank", "shortfin"]
stable_packages_list = [
    "iree-base-compiler",
    "iree-base-runtime",
    "iree-turbine",
    "sharktank",
    "shortfin",
]

if Version(PACKAGE_VERSION).is_prerelease:
    write_requirements(nightly_packages_list, PACKAGE_VERSION)
else:
    MAJOR_VERSION = Version(PACKAGE_VERSION).major
    MINOR_VERSION = Version(PACKAGE_VERSION).minor

    write_requirements(
        stable_packages_list, str(MAJOR_VERSION) + "." + str(MINOR_VERSION) + ".*"
    )
