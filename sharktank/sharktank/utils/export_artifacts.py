# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from pathlib import Path
import subprocess
import logging

import iree.compiler as ireec

logger = logging.getLogger("eval")

logger.setLevel(logging.INFO)

logger.root.handlers[0].setFormatter(
    logging.Formatter(fmt="\n%(levelname)s:%(name)-8s %(message)s")
)


class ExportArtifacts:
    def __init__(
        self,
        irpa_path: str,
        batch_size: int,
        iree_hip_target: str,
        iree_hal_target_backends: str,
        attention_kernel: str,
        tensor_parallelism_size: int,
    ):
        self.sharktank_dir = str(
            Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent
        )
        self.irpa_path = irpa_path
        self.batch_size = batch_size
        self.iree_hip_target = iree_hip_target
        self.iree_hal_target_backends = iree_hal_target_backends
        self.attention_kernel = attention_kernel
        self.tensor_parallelism_size = tensor_parallelism_size

    def export_to_mlir(
        self,
        mlir_path: str,
        json_path: str,
    ):
        export_args = [
            "python3",
            "-m",
            "sharktank.examples.export_paged_llm_v1",
            "--irpa-file",
            str(self.irpa_path),
            "--output-mlir",
            mlir_path,
            "--output-config",
            json_path,
            "--bs",
            str(self.batch_size),
        ]
        if self.attention_kernel == "decomposed":
            export_args.append("--attention-kernel")
            export_args.append(self.attention_kernel)
        elif self.attention_kernel == "torch_sdpa":
            raise NotImplementedError("attention_kernel torch_sdpa not implemented yet")
        if self.tensor_parallelism_size:
            export_args.append("--tensor-parallelism-size")
            export_args.append(str(self.tensor_parallelism_size))

        cmd = subprocess.list2cmdline(export_args)

        logger.info(
            f"export_args: {export_args}\n self.sharktank_dir: {self.sharktank_dir}"
        )

        cwd = self.sharktank_dir + "/sharktank"

        logger.debug(f"Exporting mlir:\n" f"cd {cwd} && {cmd}")
        proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd)
        return_code = proc.returncode
        if return_code != 0:
            logger.error("Error exporting mlir: ", return_code)
        else:
            logger.info(f"Exported to mlir successfully: {mlir_path}")

    def compile_to_vmfb(
        self,
        mlir_path,
        vmfb_path,
    ):
        compile_flags = ["--iree-hip-target=" + self.iree_hip_target]

        try:
            ireec.compile_file(
                input_file=mlir_path,
                target_backends=[self.iree_hal_target_backends],
                extra_args=compile_flags,
                output_file=vmfb_path,
            )
        except Exception as error:
            logger.error("Error running iree-compile: ", error)

        logger.info(f"Compiled to vmfb successfully: {vmfb_path}")

    def create_file(self, suffix, prefix):
        file_path = Path(prefix).with_suffix(suffix)
        f = open(file_path, "w")
        return file_path

    def get_artifacts(self):

        self.dir_path = self.sharktank_dir + "/" + "tmp_perplexity_ci_artifacts/"
        temp_dir = Path(self.dir_path)
        temp_dir.mkdir(parents=True, exist_ok=True)

        model_name = (
            str(self.irpa_path).split("/")[-1].split(".")[0]
            + "_"
            + self.attention_kernel
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

        if self.attention_kernel == "decomposed":
            self.export_to_mlir(
                mlir_path=mlir_path,
                json_path=json_path,
            )

            self.compile_to_vmfb(
                mlir_path=mlir_path,
                vmfb_path=vmfb_path,
            )

        return vmfb_path
