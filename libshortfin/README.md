# libshortfin - SHARK C++ inference library

## Native Dev Builds

```
cmake -GNinja -S. -Bbuild \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_LINKER_TYPE=LLD
```

## Python Dev Builds

If using a Python based development flow, there are two options:

1. `pip install -e` based.
2. Build with cmake as above and `-DSHORTFIN_BUILD_PYTHON_BINDINGS=ON` and then
   from the `build/` directory, run `pip install -v -e .` to create an
   editable install that will update as you build the C++ project.

If predominantly developing with a C++ based flow, the second option is
recommended. Your python install should track any source file changes or
builds without further interaction. Re-installing will be necessary if package
structure changes significantly.

For pure Python based dev, everything can be done from pip:

```
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

# Optional env vars:
#   SHORTFIN_IREE_SOURCE_DIR=$(pwd)/../../iree
# Note that the `--no-build-isolation` flag is useful in development setups
# because it does not create an intermediate venv that will keep later
# invocations of cmake/ninja from working at the command line. If just doing
# a one-shot build, it can be ommitted.
SHORTFIN_DEV_MODE=ON pip install --no-build-isolation -v -e .

Once built the first time, `cmake`, `ninja`, and `ctest` commands can be run
directly from `build/cmake` and changes will apply directly to the next
process launch.
```

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

```
pytest tests/
```

Run tests including for a specific platform:

```
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
