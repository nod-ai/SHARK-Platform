# Example Tuner Test

Example of tuning a dispatch and full model.

## Environments
Follow instructions in [`/tuner/README.md`](../README.md)

## Running the Tuner

### Choose a model to tune
This example uses the simple `double_mmt.mlir` file.

### Generate a benchmark file
Use the usual `iree-compile` command for your model and add
`--iree-hal-dump-executable-files-to=dump --iree-config-add-tuner-attributes`. For example:
```shell
iree-compile double_mmt.mlir --iree-hal-target-backends=rocm --iree-hip-target=gfx942 --iree-hal-dump-executable-files-to=dump --iree-config-add-tuner-attributes -o /dev/null
```

Next, copy the `*_benchmark.mlir` file from the `dump` directory to some temporary directory of choice.
This will be the input to the dispatch tuner along with the original `double_mmt.mlir` model file.

### Recommended Trial Run
For an initial trial to test the tuning loop, use:
```shell
python -m examples.test double_mmt.mlir 5 3 mmt_benchmark.mlir --num-candidates=30
```

### Basic Usage
```shell
python -m examples.test <model_file_path> <num_dispatch_candidates> <num_model_candidates> <benchmark_file_path>
```
