# SD Server and CLI

This directory contains a SD inference server, CLI and support components.


## Quick start

In your shortfin environment,
```
pip install transformers
pip install dataclasses-json
pip install pillow

```
```
python -m shortfin_apps.sd.server --help
```

## Run tests

 - From SHARK-Platform/shortfin:
 ```
 pytest --system=amdgpu -k "sd"
 ```
 The tests run with splat weights.


## Run on MI300x

 - Follow quick start

 - Navigate to shortfin/ (only necessary if you're using following CLI exactly.)
```
cd shortfin/
```
 - Run CLI server interface (you can find `sdxl_config_i8.json` in shortfin_apps/sd/examples):

The server will prepare runtime artifacts for you.

```
python -m shortfin_apps.sd.server --device=amdgpu --device_ids=0 --build_preference=precompiled --topology="spx_single"
```

 - Run a CLI client in a separate shell:
```
python -m shortfin_apps.sd.simple_client --interactive --save
```
