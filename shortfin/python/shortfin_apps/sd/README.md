# SD Server and CLI

This directory contains a SD inference server, CLI and support components.


## Quick start

Currently, we use the diffusers library for SD schedulers.
In your shortfin environment,
```
pip install diffusers@git+https://github.com/nod-ai/diffusers@0.29.0.dev0-shark
pip install transformers

```
```
python -m shortfin_apps.sd.server --help
```
