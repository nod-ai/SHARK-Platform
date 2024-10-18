# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import distutils.command.build
from pathlib import Path

from setuptools import setup  # type: ignore

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent.parent
VERSION_INFO_FILE = REPO_DIR / "version_info.json"


def load_version_info():
    with open(VERSION_INFO_FILE, "rt") as f:
        return json.load(f)


version_info = load_version_info()
PACKAGE_VERSION = version_info["package-version"]

with open(os.path.join(REPO_DIR, "README.md"), "rt") as f:
    README = f.read()


# Override build command so that we can build into _python_build
# instead of the default "build". This avoids collisions with
# typical CMake incantations, which can produce all kinds of
# hilarity (like including the contents of the build/lib directory).
class BuildCommand(distutils.command.build.build):
    def initialize_options(self):
        distutils.command.build.build.initialize_options(self)
        self.build_base = "_python_build"


setup(
    name=f"shark-platform",
    version=f"{PACKAGE_VERSION}",
    author="SHARK Authors",
    description="SHARK Platform meta package",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/nod-ai/SHARK-Platform",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    packages=[],
    install_requires=[
        # "sharktank",
        "shortfin",
    ],
    extras_require={
        "apps": [
            # TODO: add others here, or just rely on what sharktank depends on?
            #       e.g. transformers, huggingface-hub
            "dataclasses-json",
            "tokenizers",
        ],
        "onnx": [
            "iree-runtime",
            "iree-compiler[onnx]",
        ],
        "torch": [
            # TODO: plumb through [cpu,cuda,rocm] extras (if possible)
            # PyTorch uses `--index-url https://download.pytorch.org/whl/cpu`,
            # see https://pytorch.org/get-started/locally/.
            # TODO: or just drop this? if sharktank always pulls it in
            "iree-turbine",
        ],
        "serving": [
            "fastapi",
            "uvicorn",
        ],
    },
    cmdclass={"build": BuildCommand},
)
