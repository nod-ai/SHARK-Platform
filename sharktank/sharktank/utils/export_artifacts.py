# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import subprocess
import logging
import time
from pathlib import Path
from datetime import timedelta

import iree.compiler as ireec

logger = logging.getLogger("eval")

logger.setLevel(logging.INFO)

logger.root.handlers[0].setFormatter(
    logging.Formatter(fmt="\n%(levelname)s:%(name)-8s %(message)s")
)


class ExportArtifacts:
    def __init__(
        self,
        *,
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

    def timeit(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            seconds = end - start
            time_taken = abs(timedelta(seconds=round(seconds)))

            if seconds < 1:
                time_taken = f" {seconds * 1000} ms"

            func_name = func.__name__
            logger.info(f" {func_name}: {time_taken}")
            return result

        return wrapper

    @timeit
    def shard_irpa_file(
        self,
        *,
        output_file: str,
    ):
        shard_irpa_args = [
            "python3",
            "-m",
            "sharktank.models.llama.tools.shard_llama",
            "--irpa-file",
            self.irpa_path,
            "--output-file",
            output_file,
            "--shard_count",
            str(self.tensor_parallelism_size),
        ]

        cwd = self.sharktank_dir
        cmd = subprocess.list2cmdline(shard_irpa_args)

        logger.info(f"Sharding irpa file:\n" f"cd {cwd} && {cmd}")

        proc = subprocess.run(shard_irpa_args, shell=True, capture_output=True, cwd=cwd)
        if proc.returncode != 0:
            logger.error(
                f"Error sharding irpa file with shard_llama.py\n"
                f"{proc.stdout+proc.stderr}"
            )
        else:
            logger.info(f"Sharded irpa file successfully:\n" f"{proc.stdout}")

        return proc.returncode

    @timeit
    def export_to_mlir(
        self,
        *,
        mlir_path: str,
        json_path: str,
    ):
        export_args = [
            "python3",
            "-m",
            "sharktank.examples.export_paged_llm_v1",
            "--irpa-file",
            self.irpa_path,
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

        cwd = self.sharktank_dir
        cmd = subprocess.list2cmdline(export_args)

        logger.info(f"Exporting mlir:\n" f"cd {cwd} && {cmd}")

        proc = subprocess.run(export_args, capture_output=True, cwd=cwd, text=True)
        if proc.returncode != 0:
            logger.error(
                f"Error exporting mlir with export_paged_llm_v1.py\n"
                f"{proc.stdout+proc.stderr}"
            )
        else:
            logger.info(f"Exported to mlir successfully:\n" f"{proc.stdout}")

        return proc.returncode

    @timeit
    def compile_to_vmfb(
        self,
        mlir_path,
        vmfb_path,
    ):
        # TODO: Control flag to enable multiple backends
        compile_flags = ["--iree-hip-target=" + self.iree_hip_target]

        try:
            ireec.compile_file(
                input_file=mlir_path,
                target_backends=[self.iree_hal_target_backends],
                extra_args=compile_flags,
                output_file=vmfb_path,
            )
        except Exception as error:
            logger.error(f"Error running iree-compile:\n" f"{error}")
        else:
            logger.info(f"Compiled to vmfb successfully:\n" f"{vmfb_path}")

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
            returncode = self.export_to_mlir(
                mlir_path=mlir_path,
                json_path=json_path,
            )

            if returncode == 0:
                self.compile_to_vmfb(
                    mlir_path=mlir_path,
                    vmfb_path=vmfb_path,
                )

        return vmfb_path
