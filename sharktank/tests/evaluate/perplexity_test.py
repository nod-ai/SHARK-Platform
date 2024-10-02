# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import logging
import pytest

from sharktank.evaluate import perplexity

logging.basicConfig()
logger = logging.getLogger("eval")
logger.setLevel(logging.INFO)


class PerplexityTest:
    @pytest.mark.expensive
    @pytest.mark.integration
    def test(self):

        kv_cache_type = "paged"

        llama8b_f16_gguf_path = "llama8b_f16.gguf"
        llama8b_f16_tokenizer_path = "tokenizer_config.json"

        llama8b_perplexity = perplexity.main(
            [
                f"--gguf-file{llama8b_f16_gguf_path}",
                f"--tokenizer-config-json{llama8b_f16_tokenizer_path}",
                f"--kv-cache-type={kv_cache_type}",
            ]
        )

        # dummy data
        prev_llama8b_perplexity = {
            "perplexities": [
                10239.390625,
                21268.25,
                24270.857421875,
                8174.89697265625,
                11653.2939453125,
            ],
            "mean_perplexity": 15121.33779296875,
        }

        assert prev_llama8b_perplexity == llama8b_perplexity


if __name__ == "__main__":
    PerplexityTest().test()
