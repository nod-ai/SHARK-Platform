# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import pytest
import requests
import shutil
import subprocess
import sys
import time
import uuid

pytest.importorskip("transformers")
from transformers import AutoTokenizer

BATCH_SIZES = [1, 4]
cpu_settings = {
    "device_flags": [
        "-iree-hal-target-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=host",
    ],
    "device": "local-task",
}
gpu_settings = {
    "device_flags": ["-iree-hal-target-backends=rocm", "--iree-hip-target=gfx1100"],
    "device": "hip",
}
settings = cpu_settings


@pytest.fixture(scope="module")
def model_test_dir(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("cpu_llm_server_test")
    try:
        # Download model if it doesn't exist
        model_path = tmp_dir / "open-llama-3b-v2-f16.gguf"
        if not os.path.exists(model_path):
            subprocess.run(
                f"huggingface-cli download --local-dir {tmp_dir} SlyEcho/open_llama_3b_v2_gguf open-llama-3b-v2-f16.gguf",
                shell=True,
                check=True,
            )

        # Set up tokenizer if it doesn't exist
        tokenizer_path = tmp_dir / "tokenizer.json"
        if not os.path.exists(tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(
                "openlm-research/open_llama_3b_v2"
            )
            tokenizer.save_pretrained(tmp_dir)

        # Export model if it doesn't exist
        mlir_path = tmp_dir / "model.mlir"
        config_path = tmp_dir / "config.json"
        if not os.path.exists(mlir_path) or not os.path.exists(config_path):
            bs_string = ",".join(map(str, BATCH_SIZES))
            subprocess.run(
                [
                    "python",
                    "-m",
                    "sharktank.examples.export_paged_llm_v1",
                    f"--gguf-file={model_path}",
                    f"--output-mlir={mlir_path}",
                    f"--output-config={config_path}",
                    f"--bs={bs_string}",
                ],
                check=True,
            )
        # Compile model if it doesn't exist
        vmfb_path = tmp_dir / "model.vmfb"
        if not os.path.exists(vmfb_path):
            subprocess.run(
                [
                    "iree-compile",
                    mlir_path,
                    "-o",
                    vmfb_path,
                ]
                + settings["device_flags"],
                check=True,
            )
        # Write config if it doesn't exist
        edited_config_path = tmp_dir / "edited_config.json"
        if not os.path.exists(edited_config_path):
            config = {
                "module_name": "module",
                "module_abi_version": 1,
                "max_seq_len": 2048,
                "attn_head_count": 32,
                "attn_head_dim": 100,
                "prefill_batch_sizes": BATCH_SIZES,
                "decode_batch_sizes": BATCH_SIZES,
                "transformer_block_count": 26,
                "paged_kv_cache": {"block_seq_stride": 16, "device_block_count": 256},
            }
            with open(edited_config_path, "w") as f:
                json.dump(config, f)
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)


@pytest.fixture(scope="module")
def available_port(port=8000, max_port=8100):
    import socket

    starting_port = port

    while port < max_port:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                s.close()
                return port
        except socket.error:
            port += 1

    raise IOError(f"No available ports found within range {starting_port}-{max_port}")


def wait_for_server(url, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(f"{url}/health")
            return
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    raise TimeoutError(f"Server did not start within {timeout} seconds")


@pytest.fixture(scope="module")
def llm_server(model_test_dir, available_port):
    # Start the server
    server_process = subprocess.Popen(
        [
            "python",
            "-m",
            "shortfin_apps.llm.server",
            f"--tokenizer={model_test_dir / 'tokenizer.json'}",
            f"--model_config={model_test_dir / 'edited_config.json'}",
            f"--vmfb={model_test_dir / 'model.vmfb'}",
            f"--parameters={model_test_dir / 'open-llama-3b-v2-f16.gguf'}",
            f"--device={settings['device']}",
        ]
    )
    # Wait for server to start
    wait_for_server(f"http://localhost:{available_port}")
    yield server_process
    # Teardown: kill the server
    server_process.terminate()
    server_process.wait()


def do_generate(prompt, port):
    headers = {"Content-Type": "application/json"}
    # Create a GenerateReqInput-like structure
    data = {
        "text": prompt,
        "sampling_params": {"max_tokens": 50, "temperature": 0.7},
        "rid": uuid.uuid4().hex,
        "return_logprob": False,
        "logprob_start_len": -1,
        "top_logprobs_num": 0,
        "return_text_in_logprobs": False,
        "stream": False,
    }
    print("Prompt text:")
    print(data["text"])
    BASE_URL = f"http://localhost:{port}"
    response = requests.post(f"{BASE_URL}/generate", headers=headers, json=data)
    print(f"Generate endpoint status code: {response.status_code}")
    if response.status_code == 200:
        print("Generated text:")
        data = response.text
        assert data.startswith("data: ")
        data = data[6:]
        assert data.endswith("\n\n")
        data = data[:-2]
        return data
    else:
        response.raise_for_status()


def test_llm_server(llm_server, available_port):
    # Here you would typically make requests to your server
    # and assert on the responses
    assert llm_server.poll() is None
    output = do_generate("1 2 3 4 5 ", available_port)
    print(output)
    assert output.startswith("6 7 8")
