# IREE dispatch auto-tuning scripts
`libtuner.py` is the core Python script that provides the fundamental functions for the tuning loop. It imports `candidate_gen.py` for candidate generation. To implement the full tuning loop, `libtuner.py` requires a separate Python script that uses the provided `TuningClient` API from `libtuner.py`.

## Prerequisites
[Optional] Using virtual environments:
```shell
cd tuner
python -m venv .venv
source .venv/bin/activate
```
Install python dependencies:
```shell
pip install -r requirements-tuner.txt
pip install -r requirements-dev.txt
```
Using the IREE's Python bindings:
   - Building with CMake
     ```shell
     -DIREE_BUILD_PYTHON_BINDINGS=ON \
     -DPython3_EXECUTABLE="$(which python)"
     ```
   - Set environment
      ```shell
      source ../iree-build/.env && export PYTHONPATH
      ```
For more information, refer to the [IREE documentation](https://iree.dev/building-from-source/getting-started/#python-bindings)

### Overall flow

1. Symlink all scripts and mlir/irpa files in your build dir.
   - Symlink `iree-build-dir/tools` inside `tuning`.
   - Symlink ML model MLIR and weights based on `unet.sh`.

2. Copy the attention/matmul spec as `config.mlir` in the tuning dir.

3. Temporarily comment out all the existing configs in `config.mlir`.
   - Example:
     ```mlir
     // , @match_mmt_2048x10240x1280 -> @apply_op_config
     // , @match_mmt_2048x1280x5120 -> @apply_op_config
     // , @match_mmt_2048x1280x1280 -> @apply_op_config
     ```

4. Compile a baseline unet
```shell
./unet.sh winograd unet.mlir -o unet_baseline.vmfb --iree-hal-dump-executable-files-to=dump-winograd
```

5. Find the matmul to tune and copy the `*_benchmark.mlir` file to the build dir.
```shell
cp dump-winograd/*_141_*benchmark.mlir ./141.mlir
```

6. Run the tuning script.
   - Example:
    ```shell
    python -m examples.punet 141.mlir --devices=hip://GPU-0,hip://GPU-4 --num-candidates=1024
    ```

7. Check the winner candidate in `result_summary.log`, find and copy the transform spec.

8. Paste the transform spec into the `config.mlir` and uncomment them.

9. Add the match function to the entry point in `config.mlir`
   - Example:
     ```mlir
     @match_something -> @apply_op_config
     ```
