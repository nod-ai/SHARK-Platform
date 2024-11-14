# SHARK Developer Guide

Each sub-project has its own developer guide. If you would like to work across
projects, these instructions should help you get started:

### Setup a venv

We recommend setting up a Python
[virtual environment (venv)](https://docs.python.org/3/library/venv.html).
The project is configured to ignore `.venv` directories, and editors like
VSCode pick them up by default.

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install PyTorch for your system

If no explicit action is taken, the default PyTorch version will be installed.
This will give you a current CUDA-based version, which takes longer to download
and includes other dependencies that SHARK does not require. To install a
different variant, run one of these commands first:

* *CPU:*

  ```bash
  pip install -r pytorch-cpu-requirements.txt
  ```

* *ROCM:*

  ```bash
  pip install -r pytorch-rocm-requirements.txt
  ```

* *Other:* see instructions at <https://pytorch.org/get-started/locally/>.

### Install development packages

```bash
# Install editable local projects.
pip install -r requirements.txt -e sharktank/ shortfin/

# Optionally clone and install the latest editable iree-turbine dep in deps/,
# along with nightly versions of iree-base-compiler and iree-base-runtime.
pip install -f https://iree.dev/pip-release-links.html --upgrade --pre \
  iree-base-compiler iree-base-runtime --src deps \
  -e "git+https://github.com/iree-org/iree-turbine.git#egg=iree-turbine"
```

See also: [nightly_releases.md](nightly_releases.md).

### Running tests

```bash
pytest sharktank
pytest shortfin
```

### Optional: pre-commits and developer settings

This project is set up to use the `pre-commit` tooling. To install it in
your local repo, run: `pre-commit install`. After this point, when making
commits locally, hooks will run. See https://pre-commit.com/
