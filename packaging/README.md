# Packaging

We package these projects for distribution:

Package name | PyPI URL
-- | --
shark-platform | TBD
sharktank | https://pypi.org/project/sharktank
shortfin | https://pypi.org/project/shortfin
tuner | TBD

> [!NOTE]
> TODO: diagram showing dependency relationship between these projects and
> others like `iree-turbine`, `iree-compiler`, `torch`, `transformers`, etc.

## General references

* Python Package Index (PyPI): https://pypi.org/
* Python Packaging User Guide: https://packaging.python.org/en/latest/
* Setuptools: https://setuptools.pypa.io/en/latest/index.html
* Setuptools dependency management:
  https://setuptools.pypa.io/en/latest/userguide/dependency_management.html
* pip: https://pip.pypa.io/en/stable/
* pip user guide: https://pip.pypa.io/en/stable/user_guide/
* Free-threaded CPython: https://py-free-threading.github.io/

## Packages

### shark-platform

The `shark-platform` package is a meta-package containing no code itself. This
meta-package joins together compatible versions of the other packages along
with optional extras.

### sharktank

The `sharktank` package is a pure Python package, with no native modules. As a
model development toolkit, it depends on several other packages like
`gguf`, `torch`, and `transformers`.

### shortfin

The `shortfin` package contains a mix of Python APIs and native C/C++ code. As a
lightweight serving framework, it has no required package dependencies, but it
can optionally integrate with other ecosystem projects if they are available.

### tuner

TBD

## Versioning

TBD
