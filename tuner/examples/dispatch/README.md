# Dispatch Tuner

Allows to tune a signle dispatch in isolation.

## Environments
Follow instructions in [`/tuner/README.md`](../README.md)

## Running the Dispatch Tuner

### Generate a benchmark file
Use the usual `iree-compile` command for your dispatch and add
`--iree-hal-dump-executable-files-to=dump`. Copy the `*_benchmark.mlir` file
to some temporary directory of choice. This will be the input to the dispatch tuner.

### Recommended Trial Run
For an initial trial to test the tuning loop, use:
```shell
python -m examples.dispatch benchmark.mlir --num-candidates=20
```

### Dry Run Test
To perform a dry run (no GPU required), use:
```shell
python -m examples.dispatch benchmark.mlir --num-candidates=64 --num-model-candidates=10 --dry-run
```

### Basic Usage
```shell
python -m examples.dispatch benchmark.mlir
```
