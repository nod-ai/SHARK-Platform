# shortfin - SHARK C++ inference library

Build options

1. Native C++ build
2. Local Python release build
3. Package Python release build
4. Python dev build

Prerequisites

* A modern C/C++ compiler, such as clang 18 or gcc 12
* A modern Python, such as Python 3.12

## Native C++ Builds

```bash
cmake -GNinja -S. -Bbuild \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_LINKER_TYPE=LLD
cmake --build build --target all
```

If Python bindings are enabled in this mode (`-DSHORTFIN_BUILD_PYTHON_BINDINGS=ON`),
then `pip install -e build/` will install from the build dir (and support
build/continue).

## Local Python Release Builds

```bash
pip install -v -e .
```

## Package Python Release Builds

* To build wheels for Linux using a manylinux Docker container:

    ```bash
    sudo ./build_tools/build_linux_package.sh
    ```

* To build a wheel for your host OS/arch manually:

    ```bash
    # Build shortfin.*.whl into the dist/ directory
    #   e.g. `shortfin-0.9-cp312-cp312-linux_x86_64.whl`
    python3 -m pip wheel -v -w dist .

    # Install the built wheel.
    python3 -m pip install dist/*.whl
    ```

## Python Dev Builds

```bash
# Install build system pre-reqs (since we are building in dev mode, this
# is not done for us). See source of truth in pyproject.toml:
pip install setuptools wheel

# Optionally install cmake and ninja if you don't have them or need a newer
# version. If doing heavy development in Python, it is strongly recommended
# to install these natively on your system as it will make it easier to
# switch Python interpreters and build options (and the launcher in debug/asan
# builds of Python is much slower). Note CMakeLists.txt for minimum CMake
# version, which is usually quite recent.
pip install cmake ninja

SHORTFIN_DEV_MODE=ON pip install --no-build-isolation -v -e .
```

Note that the `--no-build-isolation` flag is useful in development setups
because it does not create an intermediate venv that will keep later
invocations of cmake/ninja from working at the command line. If just doing
a one-shot build, it can be ommitted.

Once built the first time, `cmake`, `ninja`, and `ctest` commands can be run
directly from `build/cmake` and changes will apply directly to the next
process launch.

Several optional environment variables can be used with setup.py:

* `SHORTFIN_CMAKE_BUILD_TYPE=Debug` : Sets the CMAKE_BUILD_TYPE. Defaults to
  `Debug` for dev mode and `Release` otherwise.
* `SHORTFIN_ENABLE_ASAN=ON` : Enables an ASAN build. Requires a Python runtime
  setup that is ASAN clean (either by env vars to preload libraries or set
  suppressions or a dev build of Python with ASAN enabled).
* `SHORTFIN_IREE_SOURCE_DIR=$(pwd)/../../iree`
* `SHORTFIN_RUN_CTESTS=ON` : Runs `ctest` as part of the build. Useful for CI
  as it uses the version of ctest installed in the pip venv.

## Running Tests

The project uses a combination of ctest for native C++ tests and pytest. Much
of the functionality is only tested via the Python tests, using the
`_shortfin.lib` internal implementation directly. In order to run these tests,
you must have installed the Python package as per the above steps.

Which style of test is used is pragmatic and geared at achieving good test
coverage with a minimum of duplication. Since it is often much more expensive
to build native tests of complicated flows, many things are only tested via
Python. This does not preclude having other language bindings later, but it
does mean that the C++ core of the library must always be built with the
Python bindings to test the most behavior. Given the target of the project,
this is not considered to be a significant issue.

### Python tests

Run platform independent tests only:

```bash
pytest tests/
```

Run tests including for a specific platform:

```bash
pytest tests/ --system amdgpu
```

# Production Library Building

In order to build a production library, additional build steps are typically
recommended:

* Compile all deps with the same compiler/linker for LTO compatibility
* Provide library dependencies manually and compile them with LTO
* Compile dependencies with `-fvisibility=hidden`
* Enable LTO builds of libshortfin
* Set flags to enable symbol versioning

# Miscellaneous Build Topics

## Free-threaded Python

Support for free-threaded Python builds (aka. "nogil") is in progress. It
is currently being tested via dev builds of CPython 3.13 with the
`--disable-gil` option set. There are multiple ways to acquire such an
environment. If using `pyenv`, here is a way:

```
# Build a free-threaded 3.13 version.
pyenv install --debug 3.13t-dev

# Test (should print "False").
pyenv shell 3.13t-dev
python -c 'import sys; print(sys._is_gil_enabled())'
```

Further ways of installing a free-threaded CPython interpreter are documented at
[py-free-threading.github.io](https://py-free-threading.github.io/installing_cpython/).
