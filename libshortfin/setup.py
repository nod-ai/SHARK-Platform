# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from distutils.core import setup, Extension
import sys
import shutil
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

def getenv_bool(key, default_value="OFF"):
    value = os.getenv(key, default_value)
    return value.upper() in ["ON", "1", "TRUE"]

ENABLE_TRACY = getenv_bool("SHORTFIN_BUILD_TRACY", "OFF" if is_cpp_prebuilt() else "ON")
if ENABLE_TRACY:
    print(
        "*** Enabling Tracy instrumentation (disable with SHORTFIN_BUILD_TRACY=OFF)",
        file=sys.stderr,
    )
else:
    print(
        "*** Tracy instrumentation not enabled (enable with SHORTFIN_BUILD_TRACY=ON)",
        file=sys.stderr,
    )
ENABLE_TRACY_TOOLS = getenv_bool("SHORTFIN_BUILD_TRACY_TOOLS")
if ENABLE_TRACY_TOOLS:
    print("*** Enabling Tracy tools (may error if missing deps)", file=sys.stderr)
else:
    print(
        "*** Tracy tools not enabled (enable with SHORTFIN_BUILD_TRACY_TOOLS=ON)",
        file=sys.stderr,
    )

if ENABLE_TRACY and is_cpp_prebuilt():
    print("Error: Tracy instrumentation cannot be enabled in prebuilt mode.")
    sys.exit(1)

if is_cpp_prebuilt():
    print("setup.py running in pre-built mode:", file=sys.stderr)
    SOURCE_DIR = Path(CPP_PREBUILT_SOURCE_DIR)
    BINARY_DIR = Path(CPP_PREBUILT_BINARY_DIR)
else:
    print("setup.py running in cmake build mode:", file=sys.stderr)
    # setup.py is in the source directory.
    SOURCE_DIR = Path(SETUPPY_DIR)
    # Note that setuptools always builds into a "build" directory that
    # is a sibling of setup.py, so we just colonize a sub-directory of that
    # by default.
    BINARY_DIR = Path(os.path.join(SETUPPY_DIR, "build", "b", "d"))

TRACY_BINARY_DIR = Path(os.path.join(SETUPPY_DIR, "build", "b", "t"))

print(f"  SOURCE_DIR = {SOURCE_DIR}", file=sys.stderr)
print(f"  BINARY_DIR = {BINARY_DIR}", file=sys.stderr)
if ENABLE_TRACY:
    print(f"  TRACY_BINARY_DIR = {BINARY_DIR}", file=sys.stderr)

# Due to a quirk of setuptools, that package_dir map must only contain
# paths relative to the directory containing setup.py. Why? No one knows.
REL_SOURCE_DIR = SOURCE_DIR.relative_to(SETUPPY_DIR, walk_up=True)
REL_BINARY_DIR = BINARY_DIR.relative_to(SETUPPY_DIR, walk_up=True)
REL_TRACY_BINARY_DIR = BINARY_DIR.relative_to(SETUPPY_DIR, walk_up=True)


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

    def copy_extensions_to_source(self, *args, **kwargs):
        ...


def maybe_nuke_cmake_cache(cmake_build_dir):
    # From run to run under pip, we can end up with different paths to ninja,
    # which isn't great and will confuse cmake. Detect if the location of
    # ninja changes and force a cache flush.
    ninja_path = ""
    try:
        import ninja
    except ModuleNotFoundError:
        pass
    else:
        ninja_path = ninja.__file__
    expected_stamp_contents = f"{sys.executable}\n{ninja_path}"

    # In order to speed things up on CI and not rebuild everything, we nuke
    # the CMakeCache.txt file if the path to the Python interpreter changed.
    # Ideally, CMake would let us reconfigure this dynamically... but it does
    # not (and gets very confused).
    PYTHON_STAMP_FILE = os.path.join(cmake_build_dir, "python_stamp.txt")
    if os.path.exists(PYTHON_STAMP_FILE):
        with open(PYTHON_STAMP_FILE, "rt") as f:
            actual_stamp_contents = f.read()
            if actual_stamp_contents == expected_stamp_contents:
                # All good.
                return

    # Mismatch or not found. Clean it.
    cmake_cache_file = os.path.join(cmake_build_dir, "CMakeCache.txt")
    if os.path.exists(cmake_cache_file):
        print("Removing CMakeCache.txt because Python version changed", file=sys.stderr)
        os.remove(cmake_cache_file)

    # And write.
    with open(PYTHON_STAMP_FILE, "wt") as f:
        f.write(expected_stamp_contents)

def build_cmake_configuration(CMAKE_BUILD_DIR: Path, extra_cmake_args=()):
    cfg = os.getenv("SHORTFIN_CMAKE_BUILD_TYPE", "Release")

    # Configure CMake.
    os.makedirs(CMAKE_BUILD_DIR, exist_ok=True)
    maybe_nuke_cmake_cache(CMAKE_BUILD_DIR)
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
    ] + list(extra_cmake_args)

    # Only do a from-scratch configure if not already configured.
    cmake_cache_file = os.path.join(CMAKE_BUILD_DIR, "CMakeCache.txt")
    if not os.path.exists(cmake_cache_file):
        print(f"Configuring with: {cmake_args}", file=sys.stderr)
        subprocess.check_call(
            ["cmake", SOURCE_DIR] + cmake_args, cwd=CMAKE_BUILD_DIR
        )
    else:
        print(f"Not re-configing (already configured)", file=sys.stderr)

    # Build.
    subprocess.check_call(["cmake", "--build", "."], cwd=CMAKE_BUILD_DIR)
    print("Build complete.", file=sys.stderr)

class CMakeBuildPy(_build_py):
    def run(self):
        # The super-class handles the pure python build.
        super().run()

        # Only Build using cmake if not in prebuild mode.
        if is_cpp_prebuilt():
            return

        self.build_default_configuration()
        if ENABLE_TRACY:
            self.build_tracy_configuration()

    def build_default_configuration(self):
        # Build extension using cmake.
        print("*********************************", file=sys.stderr)
        print("* Building base libshortfin     *", file=sys.stderr)
        print("*********************************", file=sys.stderr)

        build_cmake_configuration(BINARY_DIR)

        # We only take _shortfin_default from the build.
        target_dir = os.path.join(
            os.path.abspath(self.build_lib), "_shortfin_default"
        )
        print(f"Building in target: {target_dir}", file=sys.stderr)
        os.makedirs(target_dir, exist_ok=True)
        print("Copying build to target.", file=sys.stderr)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(
            os.path.join(
                BINARY_DIR,
                "bindings",
                "python",
                "_shortfin_default",
            ),
            target_dir,
            symlinks=False,
        )

    def build_tracy_configuration(self):
        # Build extension using cmake.
        print("*********************************", file=sys.stderr)
        print("* Building tracy libshortfin    *", file=sys.stderr)
        print("*********************************", file=sys.stderr)

        build_cmake_configuration(TRACY_BINARY_DIR)

        # We only take _shortfin_tracy from the build.
        target_dir = os.path.join(
            os.path.abspath(self.build_lib), "_shortfin_tracy"
        )
        print(f"Building in target: {target_dir}", file=sys.stderr)
        os.makedirs(target_dir, exist_ok=True)
        print("Copying build to target.", file=sys.stderr)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(
            os.path.join(
                TRACY_BINARY_DIR,
                "bindings",
                "python",
                # TODO: We should be copying _shortfin_tracy, when we build it.
                "_shortfin_default",
            ),
            target_dir,
            symlinks=False,
        )



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
        CMakeExtension("_shortfin_default.lib")
        # TODO: Conditionally map additional native library variants.
    ],
    cmdclass={
        "build": CustomBuild,
        "build_ext": NoopBuildExtension,
        "build_py": CMakeBuildPy,
    },
)
