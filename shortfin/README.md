# shortfin - SHARK inference library and serving engine

The shortfin project is SHARK's open source, high performance inference library
and serving engine. Shortfin consists of these major components:

* The "libshortfin" inference library written in C/C++ and built on
  [IREE](https://github.com/iree-org/iree)
* Python bindings for the underlying inference library
* Example applications 'shortfin_apps' built using the python bindings

## Simple user installation

Install the latest stable version:

```bash
python -m pip install shortfin
```

## Developer guides

### Install local packages

After cloning this repository, from the `shortfin/` directory:

```bash
python -m pip install .
```

Run tests:

```bash
python -m pytest -s tests/
```

### Simple dev setup

1. Check out this repository as a sibling to [IREE](https://github.com/iree-org/iree)
   if you already have an IREE source checkout. Otherwise, a pinned version will
   be downloaded for you
2. Ensure that `python --version` reads 3.11 or higher (3.12 preferred).
3. Run `./dev_me.py` to build and install the `shortfin` Python package with both
   a tracing-enabled and default build. Run it again to do an incremental build
   and delete the `build/` directory to start over
4. Run tests with `python -m pytest -s tests/`
5. Test optional features:
   * `pip install iree-base-compiler` to run a small suite of model tests intended
     to exercise the runtime (or use a [source build of IREE](https://iree.dev/building-from-source/getting-started/#using-the-python-bindings)).
   * `pip install onnx` to run some more model tests that depend on downloading
     ONNX models
   * Run tests on devices other than the CPU with flags like:
     `--system amdgpu --compile-flags="--iree-hal-target-backends=rocm --iree-hip-target=gfx1100"`
   * Use the tracy instrumented runtime to collect execution traces:
     `export SHORTFIN_PY_RUNTIME=tracy`

Refer to the advanced build options below for other scenarios.

### Advanced build options

1. Native C++ build
2. Local Python release build
3. Package Python release build
4. Python dev build

Prerequisites

* A modern C/C++ compiler, such as clang 18 or gcc 12
* A modern Python, such as Python 3.12

#### Native C++ builds

```bash
cmake -GNinja -S. -Bbuild \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_LINKER_TYPE=LLD
cmake --build build --target all
```

If Python bindings are enabled in this mode (`-DSHORTFIN_BUILD_PYTHON_BINDINGS=ON`),
then `pip install -e build/` will install from the build dir (and support
build/continue).

#### Local Python release builds

```bash
pip install -v -e .
```

#### Package Python release builds

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

#### Python dev builds

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

#### Running tests

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

##### Python tests

Run platform independent tests only:

```bash
pytest tests/
```

Run tests including for a specific platform (in this example, a gfx1100 AMDGPU):

(note that not all tests are system aware yet and some may only run on the CPU)

```bash
pytest tests/ --system amdgpu \
    --compile-flags="--iree-hal-target-backends=rocm --iree-hip-target=gfx1100"
```

## Production library building

In order to build a production library, additional build steps are typically
recommended:

* Compile all deps with the same compiler/linker for LTO compatibility
* Provide library dependencies manually and compile them with LTO
* Compile dependencies with `-fvisibility=hidden`
* Enable LTO builds of libshortfin
* Set flags to enable symbol versioning

## Miscellaneous build topics

### Free-threaded Python

Support for free-threaded Python builds (aka. "nogil") is in progress. It
is currently being tested via dev builds of CPython 3.13 with the
`--disable-gil` option set. There are multiple ways to acquire such an
environment. If using `pyenv`, here is a way:

```bash
# Build a free-threaded 3.13 version.
pyenv install --debug 3.13t-dev

# Test (should print "False").
pyenv shell 3.13t-dev
python -c 'import sys; print(sys._is_gil_enabled())'
```

Further ways of installing a free-threaded CPython interpreter are documented at
[py-free-threading.github.io](https://py-free-threading.github.io/installing_cpython/).
