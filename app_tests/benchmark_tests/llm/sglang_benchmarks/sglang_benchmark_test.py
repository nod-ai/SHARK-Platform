# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from pathlib import Path
import pytest
import time
from unittest.mock import patch

pytest.importorskip("sglang")
from sglang import bench_serving

from .utils import SGLangBenchmarkArgs, log_jsonl_result

from integration_tests.llm.utils import wait_for_server

logger = logging.getLogger(__name__)

TOKENIZER_DIR = Path("/data/llama3.1/8b/")


@pytest.mark.parametrize(
    "request_rate",
    [1, 2, 4, 8, 16, 32],
)
def test_sglang_benchmark(request_rate, sglang_args, tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("sglang_benchmark_test")
    logger.info("Beginning SGLang benchmark test...")

    port = sglang_args
    base_url = f"http://localhost:{port}"

    # Setting a high timeout gives enough time for downloading model artifacts
    # and starting up server... Takes a little longer than shortfin.
    wait_for_server(base_url, timeout=600)

    benchmark_args = SGLangBenchmarkArgs(
        backend="sglang",
        num_prompt=10,
        base_url=f"http://localhost:{port}",
        tokenizer=TOKENIZER_DIR,
        request_rate=request_rate,
    )
    output_file = (
        tmp_dir
        / f"{benchmark_args.backend}_{benchmark_args.num_prompt}_{benchmark_args.request_rate}.jsonl"
    )
    benchmark_args.output_file = output_file

    logger.info("Running SGLang Benchmark with the following args:")
    logger.info(benchmark_args)

    try:
        start = time.time()
        with patch.object(bench_serving, "print", side_effect=logger.info):
            bench_serving.run_benchmark(
                benchmark_args.as_namespace(),
            )
        logger.info(f"Benchmark run completed in {str(time.time() - start)} seconds")
        logger.info("======== RESULTS ========")
        log_jsonl_result(benchmark_args.output_file)
    except Exception as e:
        logger.error(e)
