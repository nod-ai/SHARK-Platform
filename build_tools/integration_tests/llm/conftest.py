import json
import logging
import os
from pathlib import Path
import pytest
import requests
import shutil
import subprocess
import time

pytest.importorskip("transformers")
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def model_test_dir(request, tmp_path_factory):
    """Prepare model artifacts for starting the LLM server.

    Args:
        request (FixtureRequest): The following params are accepted:
            - repo_id (str): The Hugging Face repo ID.
            - model_file (str): The model file to download.
            - tokenizer_id (str): The tokenizer ID to download.
            - settings (dict): The settings for sharktank export.
            - batch_sizes (list): The batch sizes to use for the model.
        tmp_path_factory (TempPathFactory): Temp dir to save artifacts to.

    Yields:
        Tuple[Path, Path]: The paths to the Hugging Face home and the temp dir.
    """
    logger.info("Preparing model artifacts...")

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
        logger.info(f"Preparing model_path: {model_path}..")
        if not os.path.exists(model_path):
            logger.info(
                f"Downloading model {repo_id} {model_file} from Hugging Face..."
            )
            subprocess.run(
                f"huggingface-cli download --local-dir {hf_home} {repo_id} {model_file}",
                shell=True,
                check=True,
            )
            logger.info(f"Model downloaded to {model_path}")
        else:
            logger.info("Using cached model")

        # Set up tokenizer if it doesn't exist
        tokenizer_path = hf_home / "tokenizer.json"
        logger.info(f"Preparing tokenizer_path: {tokenizer_path}...")
        if not os.path.exists(tokenizer_path):
            logger.info(f"Downloading tokenizer {tokenizer_id} from Hugging Face...")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
            )
            tokenizer.save_pretrained(hf_home)
            logger.info(f"Tokenizer saved to {tokenizer_path}")
        else:
            logger.info("Using cached tokenizer")

        # Export model
        mlir_path = tmp_dir / "model.mlir"
        config_path = tmp_dir / "config.json"
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
                f"--gguf-file={model_path}",
                f"--output-mlir={mlir_path}",
                f"--output-config={config_path}",
                f"--bs={bs_string}",
            ],
            check=True,
        )
        logger.info(f"Model successfully exported to {mlir_path}")

        # Compile model
        vmfb_path = tmp_dir / "model.vmfb"
        logger.info(f"Compiling model to {vmfb_path}")
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
        logger.info(f"Model successfully compiled to {vmfb_path}")

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
        logger.info(f"Saving edited config to: {edited_config_path}\n")
        logger.info(f"Config: {json.dumps(config, indent=2)}")
        with open(edited_config_path, "w") as f:
            json.dump(config, f)
        logger.info("Model artifacts setup successfully")
        yield hf_home, tmp_dir
    finally:
        shutil.rmtree(tmp_dir)


@pytest.fixture(scope="module")
def available_port(port=8000, max_port=8100):
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


@pytest.fixture(scope="module")
def llm_server(request, model_test_dir, available_port):
    """Start the LLM server.

    Args:
        request (FixtureRequest): The following params are accepted:
            - model_file (str): The model file to download.
            - settings (dict): The settings for starting the server.
        model_test_dir (Tuple[Path, Path]): The paths to the Hugging Face home and the temp dir.
        available_port (int): The available port to start the server on.

    Yields:
        subprocess.Popen: The server process that was started.
    """
    logger.info("Starting LLM server...")
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
