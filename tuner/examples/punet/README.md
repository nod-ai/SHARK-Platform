# Punet Tuner

## Environments
Follow instructions in [`/tuner/README.md`](../README.md)

## Shell Scripts

The required shell scripts can be downloaded from: [sdxl-scripts](https://github.com/nod-ai/sdxl-scripts)

These scripts include:
1. `compile-punet-base.sh` - Used for compiling model candidates.
2. `compile_candidate.sh` - Used for compiling dispatch candidates.
3. `punet.sh` - Invoked by `compile_candidate.sh`.

Please configure the file paths and update commands in `PunetClient`.
**Note:** Alternatively, add these scripts to your `PATH` environment variable

## Running the Tuner

### [Optional] Generate a tunable mlir
Use [`punet.sh`](https://github.com/nod-ai/sdxl-scripts/blob/main/tuning/punet.sh) to compile the sample matmul mlir [`mmt_unet.mlir`](https://github.com/nod-ai/sdxl-scripts/blob/main/tuning/mmt_unet.mlir):
```
./punet.sh ./mmt_unet.mlir -o baseline.vmfb --iree-hal-dump-executable-files-to=dump-mmt
cp ./dump-mmt/module_main_2_dispatch_0_rocm_hsaco_fb_benchmark.mlir ./2.mlir
```

### Recommended Trial Run
For an initial trial to test the tuning loop, use:
```
python punet_autotune.py 2.mlir --num-candidates=1
```

### Dry Run Test
To perform a dry run (no GPU required), use:
```
python punet_autotune.py 2.mlir --num-candidates=64 --num-model-candidates=10 --dry-run
```

### Basic Usage
```
python punet_autotune.py 2.mlir
```
