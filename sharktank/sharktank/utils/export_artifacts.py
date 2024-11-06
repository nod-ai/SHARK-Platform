# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys
import subprocess
import logging
import time
from pathlib import Path
from datetime import timedelta
from typing import List, Optional

import iree.compiler as ireec

logger = logging.getLogger("eval")

logger.setLevel(logging.INFO)

logger.root.handlers[0].setFormatter(
    logging.Formatter(fmt="\n%(levelname)s:%(name)-8s %(message)s")
)


class ExportMlirException(Exception):
    """SHARK-Platform export MLIR exception that preserves the command line and error output."""

    def __init__(self, process: subprocess.CompletedProcess, cwd: str):
        try:
            errs = process.stderr.decode("utf-8")
        except:
            errs = str(process.stderr)
        super().__init__(
            f"Error invoking export_paged_llama_v1.py\n"
            f"Error code: {process.returncode}\n"
            f"Stderr diagnostics:\n{errs}\n\n"
            f"Invoked with:\n"
            f"  cd {cwd} && {process.args}\n\n"
        )


class IreeCompileException(Exception):
    """Compiler exception that preserves the command line and error output."""

    def __init__(self, process: subprocess.CompletedProcess, cwd: str):
        try:
            errs = process.stderr.decode("utf-8")
        except:
            errs = str(process.stderr)
        super().__init__(
            f"Error invoking iree-compile\n"
            f"Error code: {process.returncode}\n"
            f"Stderr diagnostics:\n{errs}\n\n"
            f"Invoked with:\n"
            f"  cd {cwd} && {process.args}\n\n"
        )


class IreeBenchmarkException(Exception):
    """Runtime exception that preserves the command line and error output."""

    def __init__(self, process: subprocess.CompletedProcess, cwd: str):
        # iree-run-module sends output to both stdout and stderr
        try:
            errs = process.stderr.decode("utf-8")
        except:
            errs = str(process.stderr)
        try:
            outs = process.stdout.decode("utf-8")
        except:
            outs = str(process.stdout)
        super().__init__(
            f"Error invoking iree-benchmark-module\n"
            f"Error code: {process.returncode}\n"
            f"Stderr diagnostics:\n{errs}\n"
            f"Stdout diagnostics:\n{outs}\n"
            f"Run with:\n"
            f"  cd {cwd} && {process.args}\n\n"
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
        gguf_file: str,
        output_irpa: str,
    ):
        shard_irpa_args = [
            "python3",
            "-m",
            "sharktank.examples.sharding.shard_llm_dataset",
            "--gguf-file",
            gguf_file,
            "--output-irpa-file",
            output_irpa,
            "--tensor-parallelism-size",
            str(self.tensor_parallelism_size),
        ]

        cwd = self.sharktank_dir
        cmd = subprocess.list2cmdline(shard_irpa_args)

        logger.info(f"Sharding irpa file:\n" f"cd {cwd} && {cmd}")

        proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd, text=True)
        if proc.returncode != 0:
            logger.error(
                f"Error sharding irpa file with shard_llm_dataset.py\n"
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
            f"--irpa-file={self.irpa_path}",
            f"--output-mlir={mlir_path}",
            f"--output-config={json_path}",
            f"--bs={str(self.batch_size)}",
        ]
        if self.attention_kernel in ["decomposed", "torch"]:
            export_args.append("--attention-kernel")
            export_args.append(self.attention_kernel)

        cwd = self.sharktank_dir
        cmd = subprocess.list2cmdline(export_args)

        logger.info(f"Exporting mlir:\n" f"cd {cwd} && {cmd}")

        proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd, text=True)
        if proc.returncode != 0:
            raise ExportMlirException(proc, cwd)
        else:
            logger.info(f"Exported to mlir successfully:\n" f"{proc.stdout}")

        return proc.returncode

    @timeit
    def compile_to_vmfb(
        self,
        *,
        mlir_path,
        vmfb_path,
        cwd,
        hal_dump_path: Optional[Path] = None,
    ):
        # TODO: Control flag to enable multiple backends
        compile_args = [
            f"iree-compile",
            f"{mlir_path}",
            f"--iree-hip-target={self.iree_hip_target}",
            f"--iree-hal-target-backends={self.iree_hal_target_backends}",
            f"-o={vmfb_path}",
        ]
        if self.tensor_parallelism_size > 1:
            iree_hal_target_devices = [
                f"--iree-hal-target-device=hip[{i}]"
                for i in range(self.tensor_parallelism_size)
            ]
            compile_args += iree_hal_target_devices
        if hal_dump_path:
            compile_args += [
                f"--iree-hal-dump-executable-files-to={hal_dump_path}/files"
            ]

        cmd = subprocess.list2cmdline(compile_args)

        logging.getLogger().info(f"Launching compile command:\n" f"cd {cwd} && {cmd}")
        proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd)
        return_code = proc.returncode
        if return_code != 0:
            raise IreeCompileException(proc, cwd)

    def iree_benchmark_vmfb(
        self,
        *,
        hip_device_id: str,
        vmfb_name: str,
        irpa_path: str,
        args: List[str],
        cwd: str | Path,
    ):
        """Runs a compiled program with the given args using `iree-benchmark-module`.
        This assumes that the `iree-benchmark-module` command is available (usually via PATH).
        Args:
            vmfb_name: Name of the .vmfb file (relative to `cwd`).
            args: List of arguments to pass to `iree-benchmark-module`.
            cwd: Working directory to run the command within. (either string or Path works)
            compile_cmd: Command used to compile the program, for inclusion in error messages.
        Raises Exception if running fails for some reason.
        """
        benchmark_args = [
            f"ROCR_VISIBLE_DEVICES={hip_device_id}",
            "iree-benchmark-module",
            "--hip_use_streams=true",
            "--hip_allow_inline_execution=true",
            "--device_allocator=caching",
            f"--module={vmfb_name}",
        ]
        if self.tensor_parallelism_size > 1:
            import pdb

            pdb.set_trace()
            base_irpa_path, _ = os.path.splitext(irpa_path)
            params = [
                f"--parameters=model={base_irpa_path}.rank{i}.irpa"
                for i in range(self.tensor_parallelism_size)
            ]
            devices = [
                f"--device=hip[{i}]" for i in range(self.tensor_parallelism_size)
            ]
        else:
            params = [f"--parameters=model={irpa_path}"]
            devices = [f"--device=hip://{hip_device_id}"]
        benchmark_args += params
        benchmark_args += devices
        benchmark_args += args
        cmd = subprocess.list2cmdline(benchmark_args)
        logging.getLogger().info(f"Launching run command:\n" f"cd {cwd} && {cmd}")
        proc = subprocess.run(cmd, shell=True, stdout=sys.stdout, cwd=cwd)
        return_code = proc.returncode
        if return_code != 0:
            raise IreeBenchmarkException(proc, cwd)

    def create_file(self, *, suffix, prefix):
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
                    cwd=self.sharktank_dir,
                )

        return vmfb_path
