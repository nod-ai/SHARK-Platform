# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from distutils.core import setup, Extension
import os
from pathlib import Path
from setuptools.command.build_ext import build_ext as _build_ext


# This file can be generated into the build directory to allow an arbitrary
# CMake built version of the project to be installed into a venv for development.
# This can be detected if the CPP_PREBUILT global contains the string
# "TRUE", which will be the case if generated.
CPP_PREBUILT = "@SHORTFIN_PYTHON_CPP_PREBUILT@"
CPP_PREBUILT_SOURCE_DIR = "@libshortfin_SOURCE_DIR@"
CPP_PREBUILT_BINARY_DIR = "@libshortfin_BINARY_DIR@"


def is_cpp_prebuilt():
    return CPP_PREBUILT == "TRUE"


def native_build():
    if is_cpp_prebuilt():
        print("setup.py running in pre-built mode from:")
        print(f"  SOURCE_DIR = {CPP_PREBUILT_SOURCE_DIR}")
        print(f"  BINARY_DIR = {CPP_PREBUILT_BINARY_DIR}")
        return Path(CPP_PREBUILT_SOURCE_DIR), Path(CPP_PREBUILT_BINARY_DIR)
    raise RuntimeError("Packaging currently only supported in pre-built mode")


source_dir, binary_dir = native_build()

# Due to a quirk of setuptools, that package_dir map must only contain
# paths relative to the directory containing setup.py. Why? No one knows.
current_dir = Path(__file__).resolve().parent
rel_source_dir = source_dir.relative_to(current_dir, walk_up=True)
rel_binary_dir = binary_dir.relative_to(current_dir, walk_up=True)


class BuiltExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class NoopBuildExtension(_build_ext):
    def build_extension(self, ext):
        ...

    def copy_extensions_to_source(self, *args, **kwargs):
        ...


setup(
    name="libshortfin",
    version="0.9",
    description="Shortfin native library implementation",
    author="SHARK Authors",
    packages=[
        "_shortfin",
        "_shortfin_default",
        # TODO: Conditionally map additional native library variants.
    ],
    zip_safe=False,
    package_dir={
        "_shortfin": str(rel_source_dir / "bindings" / "python" / "_shortfin"),
        "_shortfin_default": str(
            rel_binary_dir / "bindings" / "python" / "_shortfin_default"
        ),
        # TODO: Conditionally map additional native library variants.
    },
    ext_modules=[
        BuiltExtension("_shortfin_default.lib"),
        # TODO: Conditionally map additional native library variants.
    ],
    cmdclass={
        "build_ext": NoopBuildExtension,
    },
)
