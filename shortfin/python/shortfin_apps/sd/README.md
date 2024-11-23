# SDXL Server and CLI

This directory contains a [SDXL](https://stablediffusionxl.com/) inference server, CLI and support components. More information about SDXL on [huggingface](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).

## Install

For [nightly releases](../../../../docs/nightly_releases.md)
For our [stable release](../../../../docs/user_guide.md)

## Start SDXL Server
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
## Run the SDXL Client

 - Run a CLI client in a separate shell:
```
python -m shortfin_apps.sd.simple_client --interactive
```
