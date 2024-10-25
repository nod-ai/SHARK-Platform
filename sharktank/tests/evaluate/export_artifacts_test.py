# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from pathlib import Path
import unittest
import pytest
import subprocess
import logging
import itertools

import iree.compiler as ireec

logger = logging.getLogger("eval")

logger.setLevel(logging.INFO)

logger.root.handlers[0].setFormatter(
    logging.Formatter(fmt="\n%(levelname)s:%(name)-8s %(message)s")
)

pytestmark = pytest.mark.usefixtures(
    "get_model_artifacts", "get_iree_flags", "tensor_parallelism_size"
)


class ExportArtifacts(unittest.TestCase):
    def setUp(self):
        self.sharktank_dir = str(
            Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
        )
        self.artifacts_dir = "/data/extra/models/"

    def export_to_mlir(
        self,
        attention_kernel: str,
        tensor_parallelism_size: int,
        irpa_path: str,
        mlir_path: str,
        json_path: str,
    ):
        export_args = [
            "python3",
            "-m",
            "sharktank.examples.export_paged_llm_v1",
            "--irpa-file",
            irpa_path,
            "--output-mlir",
            mlir_path,
            "--output-config",
            json_path,
        ]
        if attention_kernel == "decomposed":
            export_args.append("--attention-kernel")
            export_args.append(attention_kernel)
        elif self.attention_kernel == "torch_sdpa":
            raise NotImplementedError("attention_kernel torch_sdpa not implemented yet")
        if tensor_parallelism_size:
            export_args.append("--tensor-parallelism-size")
            export_args.append(str(tensor_parallelism_size))

        cmd = subprocess.list2cmdline(export_args)

        logger.info(
            f"export_args: {export_args}\n self.sharktank_dir: {self.sharktank_dir}"
        )

        logger.info(f"Exporting mlir:\n" f"cd {self.sharktank_dir} && {cmd}")
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, cwd=self.sharktank_dir
        )
        return_code = proc.returncode
        if return_code != 0:
            logger.error("Error exporting mlir: ", return_code)

    def compile_to_vmfb(
        self,
        mlir_path: str,
        vmfb_path: str,
        iree_hip_target: str,
        iree_hal_target_backends: str,
    ):
        compile_flags = ["--iree-hip-target=" + iree_hip_target]

        try:
            ireec.compile_file(
                input_file=mlir_path,
                target_backends=[iree_hal_target_backends],
                extra_args=compile_flags,
                output_file=vmfb_path,
            )
        except Exception as error:
            logger.error("Error invoking iree-compile: ", error)

    def create_file(self, suffix, prefix):
        file_path = Path(prefix).with_suffix(suffix)
        f = open(file_path, "w")
        return file_path

    def test_export(self):

        model_paths = [
            self.llama3_8b_f16_model,
            self.llama3_8b_fp8_model,
            self.llama3_405b_f16_model,
            self.llama3_405b_fp8_model,
        ]
        attention_kernels = ["decomposed", "torch_sdpa"]

        self.dir_path = self.artifacts_dir + "/" + "tmp_perplexity_ci_artifacts/"
        temp_dir = Path(self.dir_path)
        temp_dir.mkdir(parents=True, exist_ok=True)

        for model_path, attention_kernel in list(
            itertools.product(model_paths, attention_kernels)
        ):
            model_name = (
                str(model_path).split("/")[-1].split(".")[0] + "_" + attention_kernel
            )
            mlir_path = str(
                self.create_file(suffix=".mlir", prefix=self.dir_path + model_name)
            )
            json_path = str(
                self.create_file(suffix=".json", prefix=self.dir_path + model_name)
            )
            vmfb_path = str(
                self.create_file(suffix=".vmfb", prefix=self.dir_path + model_name)
            )

            if attention_kernel == "decomposed":
                self.export_to_mlir(
                    attention_kernel=attention_kernel,
                    tensor_parallelism_size=self.tensor_parallelism_size,
                    irpa_path=model_path,
                    mlir_path=mlir_path,
                    json_path=json_path,
                )

                self.compile_to_vmfb(
                    mlir_path=mlir_path,
                    vmfb_path=vmfb_path,
                    iree_hip_target=self.iree_hip_target,
                    iree_hal_target_backends=self.iree_hal_target_backends,
                )


if __name__ == "__main__":
    unittest.main()
