# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from distutils.core import setup, Extension
import sys
import subprocess
import os
from pathlib import Path
from distutils.command.build import build as _build
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py


# This file can be generated into the build directory to allow an arbitrary
# CMake built version of the project to be installed into a venv for development.
# This can be detected if the CPP_PREBUILT global contains the string
# "TRUE", which will be the case if generated.
CPP_PREBUILT = "@SHORTFIN_PYTHON_CPP_PREBUILT@"
CPP_PREBUILT_SOURCE_DIR = "@libshortfin_SOURCE_DIR@"
CPP_PREBUILT_BINARY_DIR = "@libshortfin_BINARY_DIR@"

SETUPPY_DIR = os.path.realpath(os.path.dirname(__file__))


def is_cpp_prebuilt():
    return CPP_PREBUILT == "TRUE"


if is_cpp_prebuilt():
    print("setup.py running in pre-built mode:")
    SOURCE_DIR = Path(CPP_PREBUILT_SOURCE_DIR)
    BINARY_DIR = Path(CPP_PREBUILT_BINARY_DIR)
else:
    print("setup.py running in cmake build mode:")
    # setup.py is in the source directory.
    SOURCE_DIR = Path(SETUPPY_DIR)
    BINARY_DIR = Path(os.path.join(SETUPPY_DIR, "build", "b"))

print(f"  SOURCE_DIR = {SOURCE_DIR}")
print(f"  BINARY_DIR = {BINARY_DIR}")

# Due to a quirk of setuptools, that package_dir map must only contain
# paths relative to the directory containing setup.py. Why? No one knows.
REL_SOURCE_DIR = SOURCE_DIR.relative_to(SETUPPY_DIR, walk_up=True)
REL_BINARY_DIR = BINARY_DIR.relative_to(SETUPPY_DIR, walk_up=True)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CustomBuild(_build):
    def run(self):
        self.run_command("build_py")
        self.run_command("build_ext")
        self.run_command("build_scripts")


class NoopBuildExtension(_build_ext):
    def build_extension(self, ext):
        ...

    def copy_extensions_to_source(self) -> None:
        ...


class CMakeBuildPy(_build_py):
    def run(self):
        # The super-class handles the pure python build.
        super().run()

        # Build using cmake if not in prebuild mode.
        if not is_cpp_prebuilt():

            # Build extension using cmake.
            print("*****************************", file=sys.stderr)
            print("* Building Shortfin         *", file=sys.stderr)
            print("*****************************", file=sys.stderr)

            cfg = os.getenv("SHORTFIN_CMAKE_BUILD_TYPE", "Release")

            CMAKE_BUILD_DIR = BINARY_DIR

            # Configure CMake.
            os.makedirs(BINARY_DIR, exist_ok=True)
            print(f"CMake build dir: {CMAKE_BUILD_DIR}", file=sys.stderr)
            cmake_args = [
                "-GNinja",
                "--log-level=VERBOSE",
                "-DSHORTFIN_BUNDLE_DEPS=ON",
                f"-DCMAKE_BUILD_TYPE={cfg}",
                "-DSHORTFIN_BUILD_PYTHON_BINDINGS=ON",
                # TODO: This shouldn't be hardcoded... but shortfin doesn't
                # compile without it.
                "-DCMAKE_C_COMPILER=clang",
                "-DCMAKE_CXX_COMPILER=clang++",
            ]
            print(f"Configuring with: {cmake_args}", file=sys.stderr)
            subprocess.check_call(
                ["cmake", SOURCE_DIR] + cmake_args, cwd=CMAKE_BUILD_DIR
            )

            # Build.
            subprocess.check_call(["cmake", "--build", "."], cwd=CMAKE_BUILD_DIR)
            print("Build complete.", file=sys.stderr)


PYTHON_SOURCE_DIR = REL_SOURCE_DIR / "bindings" / "python"
PYTHON_BINARY_DIR = REL_BINARY_DIR / "bindings" / "python"

# We need some directories to exist before setup.
def populate_built_package(abs_dir):
    """Makes sure that a directory and __init__.py exist.

    This needs to unfortunately happen before any of the build process
    takes place so that setuptools can plan what needs to be built.
    We do this for any built packages (vs pure source packages).
    """
    os.makedirs(abs_dir, exist_ok=True)
    with open(os.path.join(abs_dir, "__init__.py"), "wt"):
        pass


populate_built_package(os.path.join(PYTHON_BINARY_DIR / "_shortfin_default"))

setup(
    name="shortfin",
    version="0.9",
    description="Shortfin native library implementation",
    author="SHARK Authors",
    packages=[
        "_shortfin",
        "_shortfin_default",
        # TODO: Conditionally map additional native library variants.
        "shortfin",
    ],
    zip_safe=False,
    package_dir={
        "_shortfin": str(PYTHON_SOURCE_DIR / "_shortfin"),
        "_shortfin_default": str(PYTHON_BINARY_DIR / "_shortfin_default"),
        # TODO: Conditionally map additional native library variants.
        "shortfin": str(PYTHON_SOURCE_DIR / "shortfin"),
    },
    ext_modules=[
        CMakeExtension("shortfin_default.lib")
        # TODO: Conditionally map additional native library variants.
    ],
    cmdclass={
        "build": CustomBuild,
        "build_ext": NoopBuildExtension,
        "build_py": CMakeBuildPy,
    },
)
