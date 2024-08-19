# libshortfin - SHARK C++ inference library

## Dev Builds

Library dependencies:

* [spdlog](https://github.com/gabime/spdlog)
* [xtensor](https://github.com/xtensor-stack/xtensor)
* [iree runtime](https://github.com/iree-org/iree)

On recent Ubuntu, the primary dependencies can be satisfied via:

```
apt install libspdlog-dev libxtensor-dev
```

CMake must be told how to find the IREE runtime, either from a distribution
tarball, or local build/install dir. For a local build directory, pass:

```
# Assumes that the iree-build directory is adjacent to this repo.
-DCMAKE_PREFIX_PATH=$(pwd)/../../iree-build/lib/cmake/IREE
```

One liner recommended CMake command (note that CMAKE_LINKER_TYPE requires
cmake>=3.29):

```
cmake -GNinja -S. -Bbuild \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_LINKER_TYPE=LLD \
    -DCMAKE_PREFIX_PATH=$(pwd)/../../iree-build/lib/cmake/IREE
```

## Building Python Bindings

If using a Python based development flow, there are two options:

1. `pip install -v .` to build and install the library (TODO: Not yet implemented).
2. Build with cmake and `-DSHORTFIN_BUILD_PYTHON_BINDINGS=ON` and then
   from the `build/` directory, run `pip install -v -e .` to create an
   editable install that will update as you build the C++ project.

If predominantly developing with a C++ based flow, the second option is
recommended. Your python install should track any source file changes or
builds without further interaction. Re-installing will be necessary if package
structure changes significantly.

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

# Production Library Building

In order to build a production library, additional build steps are typically
recommended:

* Compile all deps with the same compiler/linker for LTO compatibility
* Provide library dependencies manually and compile them with LTO
* Compile dependencies with `-fvisibility=hidden`
* Enable LTO builds of libshortfin
* Set flags to enable symbol versioning
