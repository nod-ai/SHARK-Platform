# SHARK Modeling and Serving Libraries

> [!IMPORTANT]
> Development is still in progress for several project components. See the
> notes below for which workflows are best supported.

![GitHub License](https://img.shields.io/github/license/nod-ai/SHARK-Platform)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

<!-- TODO: high level overview, features when components are used together -->

## Sub-projects

### [`shortfin/`](./shortfin/)

<!-- TODO: features list here? -->

[![PyPI version](https://badge.fury.io/py/shortfin.svg)](https://badge.fury.io/py/shortfin) [![CI - shortfin](https://github.com/nod-ai/SHARK-Platform/actions/workflows/ci_linux_x64-libshortfin.yml/badge.svg?event=push)](https://github.com/nod-ai/SHARK-Platform/actions/workflows/ci_linux_x64-libshortfin.yml?query=event%3Apush)

The shortfin sub-project is SHARK's high performance inference library and
serving engine.

* API documentation for shortfin is available on
  [readthedocs](https://shortfin.readthedocs.io/en/latest/).

### [`sharktank/`](./sharktank/)

[![PyPI version](https://badge.fury.io/py/sharktank.svg)](https://badge.fury.io/py/sharktank) [![CI - sharktank](https://github.com/nod-ai/SHARK-Platform/actions/workflows/ci-sharktank.yml/badge.svg?event=push)](https://github.com/nod-ai/SHARK-Platform/actions/workflows/ci-sharktank.yml?query=event%3Apush)

The SHARK Tank sub-project contains a collection of model recipes and
conversion tools to produce inference-optimized programs.

> [!WARNING]
> SHARK Tank is still under development. Experienced users may want to try it
> out, but we currently recommend most users download pre-exported or
> pre-compiled model files for serving with shortfin.

<!-- TODO: features list here? -->

* See the [SHARK Tank Programming Guide](./docs/programming_guide.md) for
  information about core concepts, the development model, dataset management,
  and more.
* See [Direct Quantization with SHARK Tank](./docs/quantization.md)
  for information about quantization support.

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


## SHARK Users

If you're looking to use SHARK check out our [User Guide](docs/user_guide.md).

## SHARK Developers

If you're looking to develop SHARK, check out our [Developer Guide](docs/developer_guide.md).
