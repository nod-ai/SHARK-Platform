# SHARK Modeling and Serving Libraries

**WARNING: This is an early preview that is in progress. It is not ready for
general use.**

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)


## Development Getting Started

Use this as a guide to get started developing the project using pinned,
pre-release dependencies. You are welcome to deviate as you see fit, but
these canonical directions mirror what the CI does.

### Requirements

- Python3.12

### Setup a venv

We recommend setting up a virtual environment (venv). The project is configured
to ignore `.venv` directories, and editors like VSCode pick them up by default.

```
python -m venv --prompt sharktank .venv
source .venv/bin/activate
```

### Install PyTorch for Your System

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

### Install Development Packages

```
# Clone and install editable iree-turbine dep in deps/
pip install -f https://iree.dev/pip-release-links.html --src deps \
  -e "git+https://github.com/iree-org/iree-turbine.git#egg=iree-turbine"

# Install editable local projects.
pip install -r requirements.txt -e sharktank/ shortfin/
```

### Running Tests

```
pytest sharktank
pytest shortfin
```

### Optional: Pre-commits and developer settings

This project is set up to use the `pre-commit` tooling. To install it in
your local repo, run: `pre-commit install`. After this point, when making
commits locally, hooks will run. See https://pre-commit.com/
