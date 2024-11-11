import json
import logging
import os
from pathlib import Path
import pytest
import subprocess
import time

pytest.importorskip("sglang")

from utils import SGLangBenchmarkArgs

from integration_tests.llm.utils import (
    export_paged_llm_v1,
    compile_model,
    find_available_port,
    start_llm_server,
)

logger = logging.getLogger("__name__")

device_settings = {
    "device_flags": [
        "--iree-hal-target-backends=rocm",
        "--iree-hip-target=gfx942",
    ],
    "device": "hip",
}


def print_jsonl_output(file_path):
    with open(file_path, "r") as file:
        for line in file:
            json_object = json.loads(line.strip())
            logger.info(json.dump(json_object, indent=4))


def test_sglang_benchmark_server(tmp_path_factory):
    # TODO: Remove when multi-device is fixed
    os.environ["ROCR_VISIBLE_DEVICES"] = "1"

    tmp_dir = tmp_path_factory.mktemp("sglang_benchmark_test")

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    model_path = Path("/data/llama3.1/8b/llama8b_f16.irpa")
    mlir_path = tmp_dir / "model.mlir"
    config_path = tmp_dir / "config.json"
    vmfb_path = tmp_dir / "model.vmfb"
    tokenizer_dir = Path("/data/llama3.1/8b/")
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    batch_sizes = [1, 4]

    # Export Llama 8b f16
    if not os.path.exists(mlir_path):
        export_paged_llm_v1(mlir_path, config_path, model_path, batch_sizes)

    config = {
        "module_name": "module",
        "module_abi_version": 1,
        "max_seq_len": 2048,
        "attn_head_count": 8,
        "attn_head_dim": 100,
        "prefill_batch_sizes": batch_sizes,
        "decode_batch_sizes": batch_sizes,
        "transformer_block_count": 26,
        "paged_kv_cache": {"block_seq_stride": 16, "device_block_count": 256},
    }
    with open(config_path, "w") as file:
        json.dump(config, file)

    # Compile Model
    if not os.path.exists(vmfb_path):
        compile_model(mlir_path, vmfb_path, device_settings)

    # Start shortfin llm server
    port = find_available_port()
    server_process = start_llm_server(
        port,
        tokenizer_path,
        config_path,
        vmfb_path,
        model_path,
        device_settings,
        timeout=30,
    )

    # Run and collect SGLang Serving Benchmark
    for i in range(6):
        benchmark_args = SGLangBenchmarkArgs(
            backend="shortfin",
            num_prompt=10,
            base_url=f"http://localhost:{port}",
            tokenizer=tokenizer_dir,
            request_rate=2**i,
        )

        logger.info("Running SGLang Benchmark with the following args:")
        logger.info(benchmark_args)
        output_file = (
            tmp_dir
            / f"{benchmark_args.backend}_{benchmark_args.num_prompt}_{benchmark_args.request_rate}.jsonl"
        )
        try:
            benchmark_process = subprocess.Popen(
                [
                    "python",
                    "-m",
                    "sglang.bench_serving",
                    f"--backend={benchmark_args.backend}",
                    f"--num-prompt={benchmark_args.num_prompt}",
                    f"--base-url={benchmark_args.base_url}",
                    f"--tokenizer={benchmark_args.tokenizer}",
                    f"--request-rate={benchmark_args.request_rate}",
                    f"--output-file={output_file}",
                ],
                bufsize=-1,
            )

            timeout = 10000
            start = time.time()
            while benchmark_process.poll() is None:
                runtime = time.time() - start
                if runtime >= timeout:
                    benchmark_process.terminate()
                    benchmark_process.wait()
                    raise TimeoutError("SGLang Benchmark Timed Out")
                time.sleep(60)

            logger.info(
                f"Benchmark run completed in {str(start - time.time())} seconds"
            )
            logger.info("Test Results:")
            print_jsonl_output(output_file)
            benchmark_process.terminate()
            benchmark_process.wait()
        except Exception as e:
            logger.info(e)

    server_process.terminate()
    server_process.wait()
