# Dispatch Tuner

Allows to tune a single dispatch in isolation.

## Environments
Follow instructions in [`/tuner/README.md`](../README.md)

## Running the Dispatch Tuner

### Generate a benchmark file
Use the usual `iree-compile` command for your dispatch and add
`--iree-hal-dump-executable-files-to=dump`. For example:
```shell
iree-compile mmt.mlir --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host --iree-hal-dump-executable-files-to=dump -o /dev/null
```

Next, copy the `*_2_*_benchmark.mlir` file to some temporary directory of choice.
This will be the input to the dispatch tuner.

### Recommended Trial Run
For an initial trial to test the tuning loop, use:
```shell
python -m examples.dispatch_cpu benchmark.mlir --num-candidates=20 --device=local-task://
```

### Dry Run Test
To perform a dry run, use:
```shell
python -m examples.dispatch_cpu benchmark.mlir --num-candidates=64 --num-model-candidates=10 --device=local-task:// --dry-run
```

### Basic Usage
```shell
python -m examples.dispatch_cpu benchmark.mlir
```
