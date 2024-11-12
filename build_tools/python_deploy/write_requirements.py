#!/usr/bin/env python3
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script writes the `packaging/shark-ai/requirements.txt` file and pins
# the versions of the dependencies accordingly. For nighly releases,
#  * sharktank
#  * shortfin
# get pinned to the corresponding nighly version. For stable releases,
# * iree-base-compiler
# * iree-base-runtime
# * iree-turbine
# * sharktank
# * shortfin
# get pinned to the corresponding `X.Y.*` version.

import argparse
from pathlib import Path
import json

from packaging.version import Version


parser = argparse.ArgumentParser()
parser.add_argument("--version-suffix", action="store", type=str)

args = parser.parse_args()


THIS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = THIS_DIR.parent.parent

VERSION_FILE_SHARKTANK = REPO_ROOT / "sharktank/version_local.json"
VERSION_FILE_SHORTFIN = REPO_ROOT / "shortfin/version_local.json"
VERSION_FILE_LOCAL = REPO_ROOT / "shark-ai/version_local.json"
REQUIREMENTS_TXT = REPO_ROOT / "shark-ai/requirements.txt"


def load_version_info(version_file):
    with open(version_file, "rt") as f:
        return json.load(f)


def write_requirements(package_list, package_version):
    with open(REQUIREMENTS_TXT, "w") as f:
        for package in package_list:
            PINNED_PACKAGE = package + "==" + package_version
            f.write("%s\n" % PINNED_PACKAGE)


def append_requirements(package_list, package_version):
    with open(REQUIREMENTS_TXT, "a") as f:
        for package in package_list:
            PINNED_PACKAGE = package + "==" + package_version
            f.write("%s\n" % PINNED_PACKAGE)


metapackage_version = load_version_info(VERSION_FILE_LOCAL)
PACKAGE_VERSION = metapackage_version.get("package-version")

sharktank_version = load_version_info(VERSION_FILE_SHARKTANK)
SHARKTANK_PACKAGE_VERSION = sharktank_version.get("package-version")

shortfin_version = load_version_info(VERSION_FILE_SHORTFIN)
SHORTFIN_PACKAGE_VERSION = shortfin_version.get("package-version")

stable_packages_list = ["iree-base-compiler", "iree-base-runtime", "iree-turbine"]

if Version(PACKAGE_VERSION).is_prerelease:
    write_requirements(
        ["sharktank"],
        Version(SHARKTANK_PACKAGE_VERSION).base_version + "rc" + args.version_suffix,
    )
    append_requirements(
        ["shortfin"],
        Version(SHORTFIN_PACKAGE_VERSION).base_version + "rc" + args.version_suffix,
    )
else:
    MAJOR_VERSION = Version(PACKAGE_VERSION).major
    MINOR_VERSION = Version(PACKAGE_VERSION).minor

    write_requirements(
        stable_packages_list, str(MAJOR_VERSION) + "." + str(MINOR_VERSION) + ".*"
    )
    append_requirements(["sharktank"], Version(SHARKTANK_PACKAGE_VERSION).base_version)
    append_requirements(["shortfin"], Version(SHORTFIN_PACKAGE_VERSION).base_version)
