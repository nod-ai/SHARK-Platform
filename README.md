# SHARK Modeling and Serving Libraries

**WARNING: This is an early preview that is in progress. It is not ready for
general use.**

![GitHub License](https://img.shields.io/github/license/nod-ai/SHARK-Platform)
 [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

<!-- TODO: high level overview, features when components are used together -->

## Sub-projects

### [`sharktank/`](./sharktank/)

[![PyPI version](https://badge.fury.io/py/sharktank.svg)](https://badge.fury.io/py/sharktank) [![CI - sharktank](https://github.com/nod-ai/SHARK-Platform/actions/workflows/ci-sharktank.yml/badge.svg?event=push)](https://github.com/nod-ai/SHARK-Platform/actions/workflows/ci-sharktank.yml?query=event%3Apush)

The SHARK Tank sub-project contains a collection of model recipes and
conversion tools to produce inference-optimized programs.

<!-- TODO: features list here? -->

* See the [SHARK Tank Programming Guide](./docs/programming_guide.md) for
  information about core concepts, the development model, dataset management,
  and more.
* See [Direct Quantization with SHARK Tank](./docs/quantization.md)
  for information about quantization support.

### [`shortfin/`](./shortfin/)

<!-- TODO: features list here? -->

[![PyPI version](https://badge.fury.io/py/shortfin.svg)](https://badge.fury.io/py/shortfin) [![CI - shortfin](https://github.com/nod-ai/SHARK-Platform/actions/workflows/ci_linux_x64-libshortfin.yml/badge.svg?event=push)](https://github.com/nod-ai/SHARK-Platform/actions/workflows/ci_linux_x64-libshortfin.yml?query=event%3Apush)

The shortfin sub-project is SHARK's high performance inference library and
serving engine.

* API documentation for shortfin is available on
  [readthedocs](https://shortfin.readthedocs.io/en/latest/).

### [`tuner/`](./tuner/)

[![CI - Tuner](https://github.com/nod-ai/SHARK-Platform/actions/workflows/ci-tuner.yml/badge.svg?event=push)](https://github.com/nod-ai/SHARK-Platform/actions/workflows/ci-tuner.yml?query=event%3Apush)

The Tuner sub-project assists with tuning program performance by searching for
optimal parameter configurations to use during model compilation.

## Support matrix

<!-- TODO: version requirements for Python, ROCm, Linux, etc.  -->

### Models

Model name | Model recipes | Serving apps
---------- | ------------- | ------------
SDXL       | [`sharktank/sharktank/models/punet/`](https://github.com/nod-ai/SHARK-Platform/tree/main/sharktank/sharktank/models/punet) | [`shortfin/python/shortfin_apps/sd/`](https://github.com/nod-ai/SHARK-Platform/tree/main/shortfin/python/shortfin_apps/sd)
llama      | [`sharktank/sharktank/models/llama/`](https://github.com/nod-ai/SHARK-Platform/tree/main/sharktank/sharktank/models/llama) | [`shortfin/python/shortfin_apps/llm/`](https://github.com/nod-ai/SHARK-Platform/tree/main/shortfin/python/shortfin_apps/llm)

## Development getting started

<!-- TODO: Remove or update this section. Common setup for all projects? -->

Use this as a guide to get started developing the project using pinned,
pre-release dependencies. You are welcome to deviate as you see fit, but
these canonical directions mirror what the CI does.

### Setup a venv

We recommend setting up a virtual environment (venv). The project is configured
to ignore `.venv` directories, and editors like VSCode pick them up by default.

```
python -m venv .venv
source .venv/bin/activate
```

### Install PyTorch for your system

If no explicit action is taken, the default PyTorch version will be installed.
This will give you a current CUDA-based version. Install a different variant
by doing so explicitly first:

*CPU:*

```
pip install -r pytorch-cpu-requirements.txt
```

*ROCM:*

```
pip install -r pytorch-rocm-requirements.txt
```

### Install development packages

```
# Install editable local projects.
pip install -r requirements.txt -e sharktank/ shortfin/

# Optionally clone and install editable iree-turbine dep in deps/
pip install -f https://iree.dev/pip-release-links.html --src deps \
  -e "git+https://github.com/iree-org/iree-turbine.git#egg=iree-turbine"
```

### Running tests

```
pytest sharktank
pytest shortfin
```

### Optional: pre-commits and developer settings

This project is set up to use the `pre-commit` tooling. To install it in
your local repo, run: `pre-commit install`. After this point, when making
commits locally, hooks will run. See https://pre-commit.com/
