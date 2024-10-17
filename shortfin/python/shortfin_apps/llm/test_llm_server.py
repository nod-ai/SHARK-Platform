import pytest
import subprocess
import time
import requests
import os
import json


@pytest.fixture(scope="module")
def setup_environment():
    # Create necessary directories
    os.makedirs("/tmp/sharktank/llama", exist_ok=True)

    # Download model if it doesn't exist
    model_path = "/tmp/sharktank/llama/open-llama-3b-v2-f16.gguf"
    if not os.path.exists(model_path):
        subprocess.run(
            "huggingface-cli download --local-dir /tmp/sharktank/llama SlyEcho/open_llama_3b_v2_gguf open-llama-3b-v2-f16.gguf",
            shell=True,
            check=True,
        )

    # Set up tokenizer if it doesn't exist
    tokenizer_path = "/tmp/sharktank/llama/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        tokenizer_setup = """
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b_v2")
tokenizer.save_pretrained("/tmp/sharktank/llama")
"""
        subprocess.run(["python", "-c", tokenizer_setup], check=True)

    # Export model if it doesn't exist
    mlir_path = "/tmp/sharktank/llama/model.mlir"
    config_path = "/tmp/sharktank/llama/config.json"
    if not os.path.exists(mlir_path) or not os.path.exists(config_path):
        subprocess.run(
            [
                "python",
                "-m",
                "sharktank.examples.export_paged_llm_v1",
                f"--gguf-file={model_path}",
                f"--output-mlir={mlir_path}",
                f"--output-config={config_path}",
            ],
            check=True,
        )

    # Compile model if it doesn't exist
    vmfb_path = "/tmp/sharktank/llama/model.vmfb"
    if not os.path.exists(vmfb_path):
        subprocess.run(
            [
                "iree-compile",
                mlir_path,
                "--iree-hal-target-backends=rocm",
                "--iree-hip-target=gfx1100",
                "-o",
                vmfb_path,
            ],
            check=True,
        )

    # Write config if it doesn't exist
    edited_config_path = "/tmp/sharktank/llama/edited_config.json"
    if not os.path.exists(edited_config_path):
        config = {
            "module_name": "module",
            "module_abi_version": 1,
            "max_seq_len": 2048,
            "attn_head_count": 32,
            "attn_head_dim": 100,
            "prefill_batch_sizes": [4],
            "decode_batch_sizes": [4],
            "transformer_block_count": 26,
            "paged_kv_cache": {"block_seq_stride": 16, "device_block_count": 256},
        }
        with open(edited_config_path, "w") as f:
            json.dump(config, f)


@pytest.fixture(scope="module")
def llm_server(setup_environment):
    # Start the server
    server_process = subprocess.Popen(
        [
            "python",
            "-m",
            "shortfin_apps.llm.server",
            "--tokenizer=/tmp/sharktank/llama/tokenizer.json",
            "--model_config=/tmp/sharktank/llama/edited_config.json",
            "--vmfb=/tmp/sharktank/llama/model.vmfb",
            "--parameters=/tmp/sharktank/llama/open-llama-3b-v2-f16.gguf",
            "--device=hip",
        ]
    )

    # Wait for server to start
    time.sleep(5)

    yield server_process

    # Teardown: kill the server
    server_process.terminate()
    server_process.wait()


def test_llm_server(llm_server):
    # Here you would typically make requests to your server
    # and assert on the responses
    # For example:
    # response = requests.post("http://localhost:8000/generate", json={"prompt": "Hello, world!"})
    # assert response.status_code == 200
    # assert "generated_text" in response.json()

    # For now, we'll just check if the server process is running
    assert llm_server.poll() is None
