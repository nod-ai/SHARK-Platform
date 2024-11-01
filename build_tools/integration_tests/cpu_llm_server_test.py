# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
from pathlib import Path
import pytest
import requests
import shutil
import subprocess
import time
import uuid

pytest.importorskip("transformers")
from transformers import AutoTokenizer

CPU_SETTINGS = {
    "device_flags": [
        "-iree-hal-target-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=host",
    ],
    "device": "local-task",
}
IREE_HIP_TARGET = os.environ.get("IREE_HIP_TARGET", "gfx1100")
gpu_settings = {
    "device_flags": [
        "-iree-hal-target-backends=rocm",
        f"--iree-hip-target={IREE_HIP_TARGET}",
    ],
    "device": "hip",
}


@pytest.fixture(scope="module")
def model_test_dir(request, tmp_path_factory):
    repo_id = request.param["repo_id"]
    model_file = request.param["model_file"]
    tokenizer_id = request.param["tokenizer_id"]
    settings = request.param["settings"]
    batch_sizes = request.param["batch_sizes"]

    tmp_dir = tmp_path_factory.mktemp("cpu_llm_server_test")
    hf_home = os.environ.get("HF_HOME", None)
    hf_home = Path(hf_home) if hf_home is not None else tmp_dir
    try:
        # Download model if it doesn't exist
        model_path = hf_home / model_file
        if not os.path.exists(model_path):
            subprocess.run(
                f"huggingface-cli download --local-dir {hf_home} {repo_id} {model_file}",
                shell=True,
                check=True,
            )

        # Set up tokenizer if it doesn't exist
        tokenizer_path = hf_home / "tokenizer.json"
        if not os.path.exists(tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
            )
            tokenizer.save_pretrained(hf_home)

        # Export model if it doesn't exist
        mlir_path = tmp_dir / "model.mlir"
        config_path = tmp_dir / "config.json"
        bs_string = ",".join(map(str, batch_sizes))
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
        config = {
            "module_name": "module",
            "module_abi_version": 1,
            "max_seq_len": 2048,
            "attn_head_count": 32,
            "attn_head_dim": 100,
            "prefill_batch_sizes": batch_sizes,
            "decode_batch_sizes": batch_sizes,
            "transformer_block_count": 26,
            "paged_kv_cache": {"block_seq_stride": 16, "device_block_count": 256},
        }
        with open(edited_config_path, "w") as f:
            json.dump(config, f)
        yield hf_home, tmp_dir
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
def llm_server(request, model_test_dir, available_port):
    # Start the server
    hf_home, tmp_dir = model_test_dir
    model_file = request.param["model_file"]
    settings = request.param["settings"]
    server_process = subprocess.Popen(
        [
            "python",
            "-m",
            "shortfin_apps.llm.server",
            f"--tokenizer={hf_home / 'tokenizer.json'}",
            f"--model_config={tmp_dir / 'edited_config.json'}",
            f"--vmfb={tmp_dir / 'model.vmfb'}",
            f"--parameters={hf_home / model_file}",
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


@pytest.mark.parametrize(
    "model_test_dir,llm_server",
    [
        (
            {
                "repo_id": "SlyEcho/open_llama_3b_v2_gguf",
                "model_file": "open-llama-3b-v2-f16.gguf",
                "tokenizer_id": "openlm-research/open_llama_3b_v2",
                "settings": CPU_SETTINGS,
                "batch_sizes": [1, 4],
            },
            {"model_file": "open-llama-3b-v2-f16.gguf", "settings": CPU_SETTINGS},
        )
    ],
    indirect=True,
)
def test_llm_server(llm_server, available_port):
    # Here you would typically make requests to your server
    # and assert on the responses
    assert llm_server.poll() is None
    output = do_generate("1 2 3 4 5 ", available_port)
    print(output)
    assert output.startswith("6 7 8")
