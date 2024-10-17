# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import pytest

from sharktank.evaluate import perplexity

longrun = pytest.mark.skipif("not config.getoption('longrun')")


class PerplexityTest(unittest.TestCase):
    @longrun
    def test_llama3_8B_f16_decomposed(self):

        # Llama 3.1 8B decomposed

        llama_8b_f16_gguf_path = "/data/extra/models/llama3.1_8B/llama8b_f16.gguf"
        llama_8b_f16_tokenizer_path = (
            "/data/extra/models/llama3.1_8B/tokenizer_config.json"
        )

        llama_8b_perplexity = perplexity.main(
            [
                f"--gguf-file={llama_8b_f16_gguf_path}",
                f"--tokenizer-config-json={llama_8b_f16_tokenizer_path}",
            ]
        )

        baseline_llama_8b_perplexity = {
            "perplexities": [
                9.875290870666504,
                8.075149536132812,
                16.164775848388672,
                11.06580924987793,
                11.46964168548584,
                12.714613914489746,
            ],
            "mean_perplexity": 11.560880184173584,
        }

        delta = 5e-1

        self.assertAlmostEqual(
            baseline_llama_8b_perplexity["mean_perplexity"],
            llama_8b_perplexity["mean_perplexity"],
            delta=delta,
            msg=f"Perplexity is deviating more than {delta}",
        )

    @pytest.mark.xfail
    @longrun
    def test_llama3_8B_f16_non_decomposed(self):

        # Llama 3.1 8B non-decomposed

        llama_8b_f16_gguf_path = "/data/extra/models/llama3.1_8B/llama8b_f16.gguf"
        llama_8b_f16_tokenizer_path = (
            "/data/extra/models/llama3.1_8B/tokenizer_config.json"
        )

        llama_8b_perplexity = perplexity.main(
            [
                f"--gguf-file={llama_8b_f16_gguf_path}",
                f"--tokenizer-config-json={llama_8b_f16_tokenizer_path}",
                f"--attention-kernel=torch_sdpa",
            ]
        )

        # dummy data
        baseline_llama_8b_perplexity = {
            "perplexities": [
                9.875290870666504,
                8.075149536132812,
                16.164775848388672,
                11.06580924987793,
                11.46964168548584,
                12.714613914489746,
            ],
            "mean_perplexity": 11.560880184173584,
        }

        delta = 5e-1

        self.assertAlmostEqual(
            baseline_llama_8b_perplexity["mean_perplexity"],
            llama_8b_perplexity["mean_perplexity"],
            delta=delta,
            msg=f"Perplexity is deviating more than {delta}",
        )

    @pytest.mark.xfail
    @longrun
    def test_llama3_8B_fp8_decomposed(self):

        # Llama 3.1 8B decomposed

        llama_8b_fp8_gguf_path = "/data/extra/models/llama3.1_8B/llama8b_fp8.gguf"
        llama_8b_fp8_tokenizer_path = (
            "/data/extra/models/llama3.1_8B/tokenizer_config.json"
        )

        llama_8b_perplexity = perplexity.main(
            [
                f"--gguf-file={llama_8b_fp8_gguf_path}",
                f"--tokenizer-config-json={llama_8b_fp8_tokenizer_path}",
            ]
        )

        # dummy data
        baseline_llama_8b_perplexity = {
            "perplexities": [
                9.875290870666504,
                8.075149536132812,
                16.164775848388672,
                11.06580924987793,
                11.46964168548584,
                12.714613914489746,
            ],
            "mean_perplexity": 11.560880184173584,
        }

        delta = 5e-1

        self.assertAlmostEqual(
            baseline_llama_8b_perplexity["mean_perplexity"],
            llama_8b_perplexity["mean_perplexity"],
            delta=delta,
            msg=f"Perplexity is deviating more than {delta}",
        )

    @pytest.mark.xfail
    @longrun
    def test_llama3_8B_fp8_non_decomposed(self):

        # Llama 3.1 8B non-decomposed

        llama_8b_fp8_gguf_path = "/data/extra/models/llama3.1_8B/llama8b_fp8.gguf"
        llama_8b_fp8_tokenizer_path = (
            "/data/extra/models/llama3.1_8B/tokenizer_config.json"
        )

        llama_8b_perplexity = perplexity.main(
            [
                f"--gguf-file={llama_8b_fp8_gguf_path}",
                f"--tokenizer-config-json={llama_8b_fp8_tokenizer_path}",
                f"--attention-kernel=torch_sdpa",
            ]
        )

        # dummy data
        baseline_llama_8b_perplexity = {
            "perplexities": [
                9.875290870666504,
                8.075149536132812,
                16.164775848388672,
                11.06580924987793,
                11.46964168548584,
                12.714613914489746,
            ],
            "mean_perplexity": 11.560880184173584,
        }

        delta = 5e-1

        self.assertAlmostEqual(
            baseline_llama_8b_perplexity["mean_perplexity"],
            llama_8b_perplexity["mean_perplexity"],
            delta=delta,
            msg=f"Perplexity is deviating more than {delta}",
        )

    @longrun
    def test_llama3_8B_f16_tp8_decomposed(self):

        # Llama 3.1 8B decomposed

        llama_8b_f16_gguf_path = "/data/extra/models/llama3.1_8B/llama8b_f16.gguf"
        llama_8b_f16_tokenizer_path = (
            "/data/extra/models/llama3.1_8B/tokenizer_config.json"
        )

        tensor_parallelism_size = 8

        llama_8b_perplexity = perplexity.main(
            [
                f"--gguf-file={llama_8b_f16_gguf_path}",
                f"--tokenizer-config-json={llama_8b_f16_tokenizer_path}",
                f"--tensor-parallelism-size={tensor_parallelism_size}",
            ]
        )

        baseline_llama_8b_perplexity = {
            "perplexities": [
                9.875290870666504,
                8.075149536132812,
                16.164775848388672,
                11.06580924987793,
                11.46964168548584,
                12.714613914489746,
            ],
            "mean_perplexity": 11.560880184173584,
        }

        delta = 5e-1

        self.assertAlmostEqual(
            baseline_llama_8b_perplexity["mean_perplexity"],
            llama_8b_perplexity["mean_perplexity"],
            delta=delta,
            msg=f"Perplexity is deviating more than {delta}",
        )

    @longrun
    def test_llama3_405B_f16_decomposed(self):

        # Llama 3.1 405B decomposed

        llama_405b_f16_gguf_path = (
            "/data/extra/models/llama3.1_405B/llama405b_fp16.gguf"
        )
        llama_405b_f16_tokenizer_path = (
            "/data/extra/models/llama3.1_405B/tokenizer_config.json"
        )

        tensor_parallelism_size = 8

        llama_405b_perplexity = perplexity.main(
            [
                f"--gguf-file={llama_405b_f16_gguf_path}",
                f"--tokenizer-config-json={llama_405b_f16_tokenizer_path}",
                f"--tensor-parallelism-size={tensor_parallelism_size}",
            ]
        )

        baseline_llama_405b_perplexity = {
            "perplexities": [
                2.0203986167907715,
                4.045348644256592,
                4.452215671539307,
                4.009974479675293,
                5.169974327087402,
                5.516016960144043,
            ],
            "mean_perplexity": 4.202321449915568,
        }

        delta = 5e-1

        self.assertAlmostEqual(
            baseline_llama_405b_perplexity["mean_perplexity"],
            llama_405b_perplexity["mean_perplexity"],
            delta=delta,
            msg=f"Perplexity is deviating more than {delta}",
        )

    @pytest.mark.xfail
    @longrun
    def test_llama3_405B_f16_non_decomposed(self):

        # Llama 3.1 405B non-decomposed

        llama_405b_f16_gguf_path = (
            "/data/extra/models/llama3.1_405B/llama405b_fp16.gguf"
        )
        llama_405b_f16_tokenizer_path = (
            "/data/extra/models/llama3.1_405B/tokenizer_config.json"
        )

        tensor_parallelism_size = 8

        llama_405b_perplexity = perplexity.main(
            [
                f"--gguf-file={llama_405b_f16_gguf_path}",
                f"--tokenizer-config-json={llama_405b_f16_tokenizer_path}",
                f"--tensor-parallelism-size={tensor_parallelism_size}",
                f"--attention-kernel=torch_sdpa",
            ]
        )

        baseline_llama_405b_perplexity = {
            "perplexities": [
                2.0203986167907715,
                4.045348644256592,
                4.452215671539307,
                4.009974479675293,
                5.169974327087402,
                5.516016960144043,
            ],
            "mean_perplexity": 4.202321449915568,
        }

        delta = 5e-1

        self.assertAlmostEqual(
            baseline_llama_405b_perplexity["mean_perplexity"],
            llama_405b_perplexity["mean_perplexity"],
            delta=delta,
            msg=f"Perplexity is deviating more than {delta}",
        )

    @pytest.mark.xfail
    @longrun
    def test_llama3_405B_fp8_decomposed(self):

        # Llama 3.1 405B decomposed

        llama_405b_fp8_gguf_path = "/data/extra/models/llama3.1_405B/llama405b_fp8.gguf"
        llama_405b_fp8_tokenizer_path = (
            "/data/extra/models/llama3.1_405B/tokenizer_config.json"
        )

        tensor_parallelism_size = 8

        llama_405b_perplexity = perplexity.main(
            [
                f"--gguf-file={llama_405b_fp8_gguf_path}",
                f"--tokenizer-config-json={llama_405b_fp8_tokenizer_path}",
                f"--tensor-parallelism-size={tensor_parallelism_size}",
            ]
        )

        baseline_llama_405b_perplexity = {
            "perplexities": [
                2.0203986167907715,
                4.045348644256592,
                4.452215671539307,
                4.009974479675293,
                5.169974327087402,
                5.516016960144043,
            ],
            "mean_perplexity": 4.202321449915568,
        }

        delta = 5e-1

        self.assertAlmostEqual(
            baseline_llama_405b_perplexity["mean_perplexity"],
            llama_405b_perplexity["mean_perplexity"],
            delta=delta,
            msg=f"Perplexity is deviating more than {delta}",
        )

    @pytest.mark.xfail
    @longrun
    def test_llama3_405B_fp8_non_decomposed(self):

        # Llama 3.1 405B non-decomposed

        llama_405b_fp8_gguf_path = "/data/extra/models/llama3.1_405B/llama405b_fp8.gguf"
        llama_405b_fp8_tokenizer_path = (
            "/data/extra/models/llama3.1_405B/tokenizer_config.json"
        )

        tensor_parallelism_size = 8

        llama_405b_perplexity = perplexity.main(
            [
                f"--gguf-file={llama_405b_fp8_gguf_path}",
                f"--tokenizer-config-json={llama_405b_fp8_tokenizer_path}",
                f"--tensor-parallelism-size={tensor_parallelism_size}",
                f"--attention-kernel=torch_sdpa",
            ]
        )

        baseline_llama_405b_perplexity = {
            "perplexities": [
                2.0203986167907715,
                4.045348644256592,
                4.452215671539307,
                4.009974479675293,
                5.169974327087402,
                5.516016960144043,
            ],
            "mean_perplexity": 4.202321449915568,
        }

        delta = 5e-1

        self.assertAlmostEqual(
            baseline_llama_405b_perplexity["mean_perplexity"],
            llama_405b_perplexity["mean_perplexity"],
            delta=delta,
            msg=f"Perplexity is deviating more than {delta}",
        )


if __name__ == "__main__":
    unittest.main()
