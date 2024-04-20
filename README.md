# SHARK Modeling and Serving Libraries

**WARNING: This is an early preview that is in progress. It is not ready for
general use.**

## Development Getting Started

Use this as a guide to get started developing the project using pinned, 
pre-release dependencies. You are welcome to deviate as you see fit, but
these canonical directions mirror what the CI does.

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

This assumes you have `SHARK-Turbine` checked out adjacent (note that for the
moment we rely on pre-release versions, so installation is a bit harder).

```
pip install -f https://iree.dev/pip-release-links.html -e ../SHARK-Turbine/core/
pip install -e sharktank
pip install -e shortfin
```

### Running Tests

```
pytest sharktank
pytest shortfin
```
