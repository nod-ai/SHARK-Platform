# LLama 8b GPU Instructions on MI300X

**NOTE: This was ran on the `mi300x-3` system**

## Setup

We will use an example with `llama_8b_f16_decomposed` in order to describe the process of exporting a model for use in the shortfin llm server with an MI300 GPU.

### Pre-Requisites
- Python >= 3.11 is recommended for this flow
    - You can check out [pyenv](https://github.com/pyenv/pyenv) as a good tool to be able to manage multiple versions of python on the same system.

### Create Virtual Environment
To start, create a new virtual environment:

```bash
python -m venv --prompt shark-ai .venv
source .venv/bin/activate
```

### Install `shark-ai`
You can install either the `latest stable` version of `shark-ai` or the `nightly` version:
#### Stable
```bash
pip install shark-ai
```

#### Nightly
```bash
pip install sharktank -f https://github.com/nod-ai/SHARK-Platform/releases/expanded_assets/dev-wheels
pip install shortfin -f https://github.com/nod-ai/SHARK-Platform/releases/expanded_assets/dev-wheels
```

#### Install dataclasses-json
TODO: This should be included in release:
```bash
pip install dataclasses-json
```

### Define a directory for export files
Create a new directory for us to export files like `model.mlir`, `model.vmfb`, etc.
```bash
mkdir $PWD/export
export EXPORT_DIR=$PWD/export
```

### Define Environment Variables
Define the following environment variables to make running this example a bit easier:

#### Model/Tokenizer Vars
This example uses the `llama8b_f16.irpa` and `tokenizer.json` files that are pre-existing on the MI300X-3 system.
You may need to change the paths for your own system.
```bash
export MODEL_PARAMS_PATH=/data/llama3.1/8b/llama8b_f16.irpa # Path to existing .irpa file, may need to change w/ system
export TOKENIZER_PATH=/data/llama3.1/8b/tokenizer.json # Path to existing tokenizer.json, may need to change w/ system
```

#### Find Available Port
You can use the following command to find an available port for your server, and save it as the `$PORT` env variable:
```bash
export PORT=$(python -c "
import socket

def find_available_port(starting_port=8000, max_port=8100):
    port = starting_port
    while port < max_port:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                s.close()
                print(port)
                return port
        except socket.error:
            port += 1
    raise IOError(f'No available ports found within range {starting_port}-{max_port}')

find_available_port()
")
echo $PORT
```

#### General Env Vars
The following env vars can be copy + pasted directly:

```bash
export MLIR_PATH=$EXPORT_DIR/model.mlir # Path to export model.mlir file
export OUTPUT_CONFIG_PATH=$EXPORT_DIR/config.json # Path to export config.json file
export EDITED_CONFIG_PATH=$EXPORT_DIR/edited_config.json # Path to export config.json file
export VMFB_PATH=$EXPORT_DIR/model.vmfb # Path to export model.vmfb file
export BS=1,4 # Batch size for kvcache
export ROCR_VISIBLE_DEVICES=1 # NOTE: This is temporary, until multi-device is fixed
```

### Export to MLIR
We will now use the `sharktank.examples.export_paged_llm_v1` script to export our model to `.mlir` format.
```bash
python -m sharktank.examples.export_paged_llm_v1 \
  --irpa-file=$MODEL_PARAMS_PATH \
  --output-mlir=$MLIR_PATH \
  --output-config=$OUTPUT_CONFIG_PATH \
  --bs=$BS
```

## Compiling to `.vmfb`

Now that we have generated a `model.mlir` file, we can compile it to `.vmfb` format, which is required for running the `shortfin` LLM server.

We will use the [iree-compile](https://iree.dev/developers/general/developer-overview/#iree-compile) tool for compiling our model.

#### Compile for MI300
**NOTE: This command is specific to MI300 GPUs. For other `--iree-hip-target` GPU options, look [here](https://iree.dev/guides/deployment-configurations/gpu-rocm/#compile-a-program)**
```bash
iree-compile $MLIR_PATH \
 --iree-hal-target-backends=rocm \
 --iree-hip-target=gfx942 \
 -o $VMFB_PATH
```

## Write an Edited Config

We need to write a config for our model with a slightly edited structure to run with shortfin. This will work for the example in our docs.
You may need to modify some of the parameters for a specific model.

### Write Edited Config

```bash
cat > $EDITED_CONFIG_PATH << EOF
{
    "module_name": "module",
    "module_abi_version": 1,
    "max_seq_len": 131072,
    "attn_head_count": 8,
    "attn_head_dim": 128,
    "prefill_batch_sizes": [
        $BS
    ],
    "decode_batch_sizes": [
        $BS
    ],
    "transformer_block_count": 32,
    "paged_kv_cache": {
        "block_seq_stride": 16,
        "device_block_count": 256
    }
}
EOF
```

## Running the `shortfin` LLM Server

We should now have all of the files that we need to run the shortfin LLM server.

Verify that you have the following in your specified directory ($EXPORT_DIR):

```bash
ls $EXPORT_DIR
```

- edited_config.json
- model.vmfb

### Launch Server:

#### Set the Target Device
TODO: Add instructions on targeting different devices, when `--device=hip://$DEVICE` is supported

#### Run the Shortfin Server
Run the following command to launch the Shortfin LLM Server in the background:
```
python -m shortfin_apps.llm.server \
   --tokenizer_json=$TOKENIZER_PATH \
   --model_config=$EDITED_CONFIG_PATH \
   --vmfb=$VMFB_PATH \
   --parameters=$MODEL_PARAMS_PATH \
   --device=hip \
   --port=$PORT > shortfin_llm_server.log 2>&1 &
shortfin_process=$!
```

You can verify your command has launched successfully when you see the following logs outputted to terminal:

```bash
cat shortfin_llm_server.log
```

#### Expected Output
```text
[2024-10-24 15:40:27.440] [info] [on.py:62] Application startup complete.
[2024-10-24 15:40:27.444] [info] [server.py:214] Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Verify Server
We can now verify our LLM server by sending a simple request:

### Open Python shell
```bash
python
```

### Send Request
```python
import requests

import os

generate_url = f"http://localhost:{os.environ['PORT']}/generate"

def generation_request():
    payload = {"text": "What is the capital of the United States?", "sampling_params": {"max_completion_tokens": 50}}
    try:
        resp = requests.post(generate_url, json=payload)
        resp.raise_for_status()  # Raises an HTTPError for bad responses
        print(resp.text)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

generation_request()
```

After you receive the request, you can exit the python shell:

```bash
quit()
```

## Cleanup
When done, you can kill the shortfin_llm_server by killing the process:
```bash
kill -9 $shortfin_process
```
