# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import distutils.command.build
from pathlib import Path

from setuptools import find_namespace_packages, setup  # type: ignore

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent
VERSION_INFO_FILE = REPO_DIR / "version_info.json"


with open(
    os.path.join(
        THIS_DIR,
        "README.md",
    ),
    "rt",
) as f:
    README = f.read()


def load_version_info():
    with open(VERSION_INFO_FILE, "rt") as f:
        return json.load(f)


version_info = load_version_info()
PACKAGE_VERSION = version_info["package-version"]

packages = find_namespace_packages(
    include=[
        "sharktank",
        "sharktank.*",
    ],
)

print("Found packages:", packages)

# Lookup version pins from requirements files.
requirement_pins = {}


def load_requirement_pins(requirements_file: Path):
    with open(requirements_file, "rt") as f:
        lines = f.readlines()
    pin_pairs = [line.strip().split("==") for line in lines if "==" in line]
    requirement_pins.update(dict(pin_pairs))


load_requirement_pins(REPO_DIR / "requirements.txt")


def get_version_spec(dep: str):
    if dep in requirement_pins:
        return f">={requirement_pins[dep]}"
    else:
        return ""


# Override build command so that we can build into _python_build
# instead of the default "build". This avoids collisions with
# typical CMake incantations, which can produce all kinds of
# hilarity (like including the contents of the build/lib directory).
class BuildCommand(distutils.command.build.build):
    def initialize_options(self):
        distutils.command.build.build.initialize_options(self)
        self.build_base = "_python_build"


setup(
    name=f"sharktank",
    version=f"{PACKAGE_VERSION}",
    author="SHARK Authors",
    author_email="stella@nod.ai",
    description="SHARK layers and inference models for genai",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/nod-ai/sharktank",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    packages=packages,
    include_package_data=True,
    package_data={
        "sharktank": ["py.typed", "kernels/templates/*.mlir"],
    },
    install_requires=[
        "iree-turbine",
    ],
    extras_require={
        "testing": [
            f"pytest{get_version_spec('pytest')}",
            f"pytest-xdist{get_version_spec('pytest-xdist')}",
        ],
    },
    cmdclass={"build": BuildCommand},
)
