# Punet Tuner

## Environments
Follow instructions in [`/tuner/README.md`](../README.md)

## Shell Scripts

The required shell scripts can be downloaded from:
[sdxl-scripts](https://github.com/nod-ai/sdxl-scripts).

These scripts include:
1. `compile-punet-base.sh` - Used for compiling model candidates.
2. `compile_candidate.sh` - Used for compiling dispatch candidates.
3. `punet.sh` - Invoked by `compile_candidate.sh`.

Add the parent directories of these scripts to your `PATH` environment variable,
so that they can be picked up by `punet_autotune.py`.

## Running the Tuner

### [Optional] Generate a tunable mlir
Use
[`punet.sh`](https://github.com/nod-ai/sdxl-scripts/blob/main/tuning/punet.sh)
to compile the sample matmul `mmt.mlir` (can also find here:
[`mmt_unet.mlir`](https://github.com/nod-ai/sdxl-scripts/blob/main/tuning/mmt_unet.mlir)):
```shell
punet.sh mmt.mlir -o mmt.vmfb --iree-hal-dump-executable-files-to=dump-mmt
cp ./dump-mmt/module_main_0_dispatch_0_rocm_hsaco_fb_benchmark.mlir test-benchmark.mlir
```

### Recommended Trial Run
For an initial trial to test the tuning loop, use:
```shell
python -m examples.punet test-benchmark.mlir --num-candidates=10
```

### Dry Run Test
To perform a dry run (no GPU required), use:
```shell
python -m examples.punet test-benchmark.mlir --num-candidates=64 --num-model-candidates=10 --dry-run
```

### Basic Usage
```shell
python -m examples.punet test-benchmark.mlir
```
