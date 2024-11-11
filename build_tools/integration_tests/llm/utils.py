# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import os
import subprocess
import time

import requests
from transformers import AutoTokenizer

logger = logging.getLogger("__name__")


class AccuracyValidationException(RuntimeError):
    pass


def download_huggingface_model(local_dir, repo_id, model_file):
    model_path = local_dir / model_file
    logger.info(f"Preparing model_path: {model_path}..")
    if not os.path.exists(model_path):
        logger.info(f"Downloading model {repo_id} {model_file} from Hugging Face...")
        subprocess.run(
            f"huggingface-cli download --local-dir {local_dir} {repo_id} {model_file}",
            shell=True,
            check=True,
        )
        logger.info(f"Model downloaded to {model_path}")
    else:
        logger.info("Using cached model")


def download_tokenizer(local_dir, tokenizer_id):
    # Set up tokenizer if it doesn't exist
    tokenizer_path = local_dir / "tokenizer.json"
    logger.info(f"Preparing tokenizer_path: {tokenizer_path}...")
    if not os.path.exists(tokenizer_path):
        logger.info(f"Downloading tokenizer {tokenizer_id} from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
        )
        tokenizer.save_pretrained(local_dir)
        logger.info(f"Tokenizer saved to {tokenizer_path}")
    else:
        logger.info("Using cached tokenizer")


def export_paged_llm_v1(mlir_path, config_path, model_path, batch_sizes):
    bs_string = ",".join(map(str, batch_sizes))
    logger.info(
        "Exporting model with following settings:\n"
        f"  MLIR Path: {mlir_path}\n"
        f"  Config Path: {config_path}\n"
        f"  Batch Sizes: {bs_string}"
    )
    subprocess.run(
        [
            "python",
            "-m",
            "sharktank.examples.export_paged_llm_v1",
            f"--{model_path.suffix.strip('.')}-file={model_path}",
            f"--output-mlir={mlir_path}",
            f"--output-config={config_path}",
            f"--bs={bs_string}",
        ],
        check=True,
    )
    logger.info(f"Model successfully exported to {mlir_path}")


def compile_model(mlir_path, vmfb_path, device_settings):
    logger.info(f"Compiling model to {vmfb_path}")
    subprocess.run(
        [
            "iree-compile",
            mlir_path,
            "-o",
            vmfb_path,
        ]
        + device_settings["device_flags"],
        check=True,
    )
    logger.info(f"Model successfully compiled to {vmfb_path}")


def find_available_port(port=8000, max_port=8100):
    import socket

    logger.info(f"Finding available port in range {port}-{max_port}...")

    starting_port = port

    while port < max_port:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                s.close()
                logger.info(f"Found available port: {port}")
                return port
        except socket.error:
            port += 1

    raise IOError(f"No available ports found within range {starting_port}-{max_port}")


def wait_for_server(url, timeout=10):
    logger.info(f"Waiting for server to start at {url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(f"{url}/health")
            logger.info("Server successfully started")
            return
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    raise TimeoutError(f"Server did not start within {timeout} seconds")


def start_llm_server(
    port,
    tokenizer_path,
    model_config_path,
    vmfb_path,
    parameters_path,
    settings,
    timeout=10,
):
    logger.info("Starting LLM server...")
    # Start the server
    server_process = subprocess.Popen(
        [
            "python",
            "-m",
            "shortfin_apps.llm.server",
            f"--tokenizer_json={tokenizer_path}",
            f"--model_config={model_config_path}",
            f"--vmfb={vmfb_path}",
            f"--parameters={parameters_path}",
            f"--device={settings['device']}",
            f"--port={port}",
        ]
    )
    # Wait for server to start
    wait_for_server(f"http://localhost:{port}", timeout)
    return server_process
