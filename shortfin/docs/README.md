# Python API Docs

Documentation for the Python API is build with Sphinx under this directory.

## Building docs

The Python modules will be automatically imported if installed or if the build
is located at `../build`, relative to this file.

### Install dependencies

```shell
python3 -m pip install -r requirements.txt
```

### Build the docs

```shell
sphinx-build -b html . _build
```
