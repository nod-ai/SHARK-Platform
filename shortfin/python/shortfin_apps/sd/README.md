# SD Server and CLI

This directory contains a SD inference server, CLI and support components.


## Quick start

In your shortfin environment,
```
pip install transformers
pip install dataclasses-json
pip install pillow
pip install shark-ai

```
```
python -m shortfin_apps.sd.server --help
```

# Run on MI300x
The server will prepare runtime artifacts for you.

By default, the port is set to 8000. If you would like to change this, use `--port` in each of the following commands.

You can check if this (or any) port is in use on Linux with `ss -ntl | grep 8000`.

```
python -m shortfin_apps.sd.server --device=amdgpu --device_ids=0 --build_preference=precompiled --topology="spx_single"
```
 - Wait until your server outputs:
```
INFO - Application startup complete.
INFO - Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
 - Run a CLI client in a separate shell:
```
python -m shortfin_apps.sd.simple_client --interactive
```
