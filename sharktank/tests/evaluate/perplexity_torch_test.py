# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import pytest
import json

from sharktank.evaluate import perplexity_torch

longrun = pytest.mark.skipif("not config.getoption('longrun')")


@pytest.mark.usefixtures("get_model_path")
class PerplexityTest(unittest.TestCase):
    def setUp(self):
        self.current_perplexity_all = {}
        self.delta = 5e-1
        self.tensor_parallelism_size = 8
        with open(self.baseline_perplexity_scores, "r") as f:
            self.baseline_perplexity = json.load(f)

    @longrun
    def test_llama3_8B_f16_decomposed(self):

        # Llama 3.1 8B decomposed

        model_name = "llama3_8B_f16_decomposed"
        baseline_perplexity = self.baseline_perplexity[model_name]

        current_perplexity = perplexity_torch.main(
            [
                f"--irpa-file={self.llama3_8b_f16_model}",
                f"--tokenizer-config-json={self.llama3_8b_tokenizer}",
            ]
        )

        perplexity_difference = (
            current_perplexity["mean_perplexity"]
            - baseline_perplexity["mean_perplexity"]
        )

        self.assertAlmostEqual(
            baseline_perplexity["mean_perplexity"],
            current_perplexity["mean_perplexity"],
            delta=self.delta,
            msg=f"Current perplexity deviates baseline by {perplexity_difference}",
        )

    @pytest.mark.xfail(
        reason="Non-decomposed attention is not supported yet",
    )
    @longrun
    def test_llama3_8B_f16_non_decomposed(self):

        # Llama 3.1 8B non-decomposed

        model_name = "llama3_8B_f16_non_decomposed"
        baseline_perplexity = self.baseline_perplexity[model_name]

        current_perplexity = perplexity_torch.main(
            [
                f"--irpa-file={self.llama3_8b_f16_model}",
                f"--tokenizer-config-json={self.llama3_8b_tokenizer}",
                f"--attention-kernel=torch_sdpa",
            ]
        )

        perplexity_difference = (
            current_perplexity["mean_perplexity"]
            - baseline_perplexity["mean_perplexity"]
        )

        self.assertAlmostEqual(
            baseline_perplexity["mean_perplexity"],
            current_perplexity["mean_perplexity"],
            delta=self.delta,
            msg=f"Current perplexity deviates baseline by {perplexity_difference}",
        )

    @pytest.mark.xfail(
        reason="FP8 model is unsupported",
    )
    @longrun
    def test_llama3_8B_fp8_decomposed(self):

        # Llama 3.1 8B decomposed

        model_name = "llama3_8B_fp8_decomposed"
        baseline_perplexity = self.baseline_perplexity[model_name]

        current_perplexity = perplexity_torch.main(
            [
                f"--irpa-file={self.llama3_8b_fp8_model}",
                f"--tokenizer-config-json={self.llama3_8b_tokenizer}",
            ]
        )

        perplexity_difference = (
            current_perplexity["mean_perplexity"]
            - baseline_perplexity["mean_perplexity"]
        )

        self.assertAlmostEqual(
            baseline_perplexity["mean_perplexity"],
            current_perplexity["mean_perplexity"],
            delta=self.delta,
            msg=f"Current perplexity deviates baseline by {perplexity_difference}",
        )

    @pytest.mark.xfail(
        reason="Non-decomposed attention is not supported yet",
    )
    @longrun
    def test_llama3_8B_fp8_non_decomposed(self):

        # Llama 3.1 8B non-decomposed

        model_name = "llama3_8B_fp8_non_decomposed"
        baseline_perplexity = self.baseline_perplexity[model_name]

        current_perplexity = perplexity_torch.main(
            [
                f"--irpa-file={self.llama3_8b_fp8_model}",
                f"--tokenizer-config-json={self.llama3_8b_tokenizer}",
                f"--attention-kernel=torch_sdpa",
            ]
        )

        perplexity_difference = (
            current_perplexity["mean_perplexity"]
            - baseline_perplexity["mean_perplexity"]
        )

        self.assertAlmostEqual(
            baseline_perplexity["mean_perplexity"],
            current_perplexity["mean_perplexity"],
            delta=self.delta,
            msg=f"Current perplexity deviates baseline by {perplexity_difference}",
        )

    @longrun
    def test_llama3_405B_f16_decomposed(self):

        # Llama 3.1 405B decomposed

        model_name = "llama3_405B_f16_decomposed"
        baseline_perplexity = self.baseline_perplexity[model_name]

        current_perplexity = perplexity_torch.main(
            [
                f"--irpa-file={self.llama3_405b_f16_model}",
                f"--tokenizer-config-json={self.llama3_405b_tokenizer}",
                f"--tensor-parallelism-size={self.tensor_parallelism_size}",
            ]
        )

        perplexity_difference = (
            current_perplexity["mean_perplexity"]
            - baseline_perplexity["mean_perplexity"]
        )

        self.assertAlmostEqual(
            baseline_perplexity["mean_perplexity"],
            current_perplexity["mean_perplexity"],
            delta=self.delta,
            msg=f"Current perplexity deviates baseline by {perplexity_difference}",
        )

    @pytest.mark.xfail(
        reason="Non-decomposed attention is not supported yet",
    )
    @longrun
    def test_llama3_405B_f16_non_decomposed(self):

        # Llama 3.1 405B non-decomposed

        model_name = "llama3_405B_f16_non_decomposed"
        baseline_perplexity = self.baseline_perplexity[model_name]

        current_perplexity = perplexity_torch.main(
            [
                f"--irpa-file={self.llama3_405b_f16_model}",
                f"--tokenizer-config-json={self.llama3_405b_tokenizer}",
                f"--tensor-parallelism-size={self.tensor_parallelism_size}",
                f"--attention-kernel=torch_sdpa",
            ]
        )

        perplexity_difference = (
            current_perplexity["mean_perplexity"]
            - baseline_perplexity["mean_perplexity"]
        )

        self.assertAlmostEqual(
            baseline_perplexity["mean_perplexity"],
            current_perplexity["mean_perplexity"],
            delta=self.delta,
            msg=f"Current perplexity deviates baseline by {perplexity_difference}",
        )

    @pytest.mark.xfail(
        reason="FP8 model is unsupported",
    )
    @longrun
    def test_llama3_405B_fp8_decomposed(self):

        # Llama 3.1 405B decomposed

        model_name = "llama3_405B_fp8_decomposed"
        baseline_perplexity = self.baseline_perplexity[model_name]

        current_perplexity = perplexity_torch.main(
            [
                f"--irpa-file={self.llama3_405b_fp8_model}",
                f"--tokenizer-config-json={self.llama3_405b_tokenizer}",
                f"--tensor-parallelism-size={self.tensor_parallelism_size}",
            ]
        )

        perplexity_difference = (
            current_perplexity["mean_perplexity"]
            - baseline_perplexity["mean_perplexity"]
        )

        self.assertAlmostEqual(
            baseline_perplexity["mean_perplexity"],
            current_perplexity["mean_perplexity"],
            delta=self.delta,
            msg=f"Current perplexity deviates baseline by {perplexity_difference}",
        )

    @pytest.mark.xfail(
        reason="Non-decomposed attention is not supported yet",
    )
    @longrun
    def test_llama3_405B_fp8_non_decomposed(self):

        # Llama 3.1 405B non-decomposed

        model_name = "llama3_405B_fp8_non_decomposed"
        baseline_perplexity = self.baseline_perplexity[model_name]

        current_perplexity = perplexity_torch.main(
            [
                f"--irpa-file={self.llama3_405b_fp8_model}",
                f"--tokenizer-config-json={self.llama3_405b_tokenizer}",
                f"--tensor-parallelism-size={self.tensor_parallelism_size}",
                f"--attention-kernel=torch_sdpa",
            ]
        )

        perplexity_difference = (
            current_perplexity["mean_perplexity"]
            - baseline_perplexity["mean_perplexity"]
        )

        self.assertAlmostEqual(
            baseline_perplexity["mean_perplexity"],
            current_perplexity["mean_perplexity"],
            delta=self.delta,
            msg=f"Current perplexity deviates baseline by {perplexity_difference}",
        )


if __name__ == "__main__":
    unittest.main()
