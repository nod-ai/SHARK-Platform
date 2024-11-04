# Nightly releases

> [!WARNING]
> This is still under development! See
> https://github.com/nod-ai/SHARK-Platform/issues/400.
>
> These instructions will be converted into a user guide once stable packages
> are published to PyPI: <https://github.com/nod-ai/SHARK-Platform/issues/359>.

Nightly releases are uploaded to
https://github.com/nod-ai/SHARK-Platform/releases/tag/dev-wheels.

* The "expanded_assets" version of a release page is compatible with the
  `-f, --find-links <url>` options of `pip install`
  ([docs here](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-f)).
  For the "dev-wheels" release above, that page is:
  <https://github.com/nod-ai/SHARK-Platform/releases/expanded_assets/dev-wheels>
* These releases are generated using
  [`.github/workflows/build_package.yml`](../.github/workflows/build_packages.yml)
* That workflow runs the
  [`sharktank/build_tools/build_linux_package.sh`](../sharktank/build_tools/build_linux_package.sh)
  and
[`shortfin/build_tools/build_linux_package.sh`](../shortfin/build_tools/build_linux_package.sh)
  scripts
* Workflow history can be viewed at
  <https://github.com/nod-ai/SHARK-Platform/actions/workflows/build_packages.yml>

## Prerequisites

### Operating system

Currently we only officially support Linux with published packages. Windows and
macOS support is possible, but may need additional setup, code changes, and
source builds.

### Python

You will need a recent version of Python.

* As of Nov 1, 2024, sharktank is compatible with Python 3.11. See
  https://github.com/nod-ai/SHARK-Platform/issues/349 for Python 3.12 support.
* As of Nov 1, 2024, shortfin publishes packages for Python 3.11, 3.12, 3.13,
  and 3.13t

For example, to install Python 3.11 on Ubuntu:

```bash
sudo apt install python3.11 python3.11-dev python3.11-venv

which python3.11
# /usr/bin/python3.11
```

> [!NOTE]
> Tip: manage multiple Python versions using `pyenv`
> (<https://github.com/pyenv/pyenv>), or `update-alternatives` on Linux
> ([guide here](https://linuxconfig.org/how-to-change-from-default-to-alternative-python-version-on-debian-linux))
> , or the
> [Python Launcher for Windows](https://docs.python.org/3/using/windows.html#python-launcher-for-windows)
> on Windows.

## Quickstart - sharktank

```bash
# Set up a virtual environment to isolate packages from other envs.
python3.11 -m venv 3.11.venv
source 3.11.venv/bin/activate

# Install 'sharktank' package from nightly releases.
python -m pip install sharktank -f https://github.com/nod-ai/SHARK-Platform/releases/expanded_assets/dev-wheels

# Install iree-turbine from source
# TODO(#294): publish newer iree-turbine package so this isn't necessary.
python -m pip install -f https://iree.dev/pip-release-links.html --src deps \
  -e "git+https://github.com/iree-org/iree-turbine.git#egg=iree-turbine"

# Install some other dependencies.
# TODO(#294): list these as "dependencies" in `pyproject.toml` or make optional?
python -m pip install gguf numpy huggingface-hub transformers datasets \
  sentencepiece

# Test the installation.
python -c "from sharktank import ops; print('Sanity check passed')"

# Deactivate the virtual environment when done.
deactivate
```

## Quickstart - shortfin

```bash
# Set up a virtual environment to isolate packages from other envs.
python3.12 -m venv 3.12.venv
source 3.12.venv/bin/activate

# Install 'shortfin' package from nightly releases.
python -m pip install shortfin -f https://github.com/nod-ai/SHARK-Platform/releases/expanded_assets/dev-wheels

# Test the installation.
python -c "import shortfin as sf; print('Sanity check passed')"

# Deactivate the virtual environment when done.
deactivate
```
