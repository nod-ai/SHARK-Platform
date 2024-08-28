# Punet Tuner

## Environments
Follow instructions in `/tuner/README.md`

## Shell Scripts

The required shell scripts can be downloaded from: [sdxl-scripts](https://github.com/nod-ai/sdxl-scripts)

These scripts include:
1. `compile-punet-base.sh` - Used for compiling model candidates.
2. `compile_candidate.sh` - Used for compiling dispatch candidates.
3. `punet.sh` - Invoked by `compile_candidate.sh`.

Please configure the file paths and update commands in `PunetClient`.

## Running the Tuner

### Recommended Trial Run
For an initial trial to test the tuning loop, use:
```
python punet_autotune.py 1286.mlir --num-candidates=1
```

### Dry Run Test
To perform a dry run (no GPU required), use:
```
python punet_autotune.py 1286.mlir --num-candidates=64 --num-model-candidates=10 --dry-run
```

### Basic Usage
```
python punet_autotune.py 1286.mlir
```