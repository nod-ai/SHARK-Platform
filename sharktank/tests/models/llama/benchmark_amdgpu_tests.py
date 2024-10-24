# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from datetime import datetime
import os
import sys
import unittest
import pytest
import subprocess
from pathlib import Path
from typing import List

longrun = pytest.mark.skipif("not config.getoption('longrun')")
is_mi300x = pytest.mark.skipif("config.getoption('iree_hip_target') != 'gfx942'")


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

    def __init__(
        self, process: subprocess.CompletedProcess, cwd: str, compile_cmd: str
    ):
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
            f"Compiled with:\n"
            f"  cd {cwd} && {compile_cmd}\n\n"
            f"Run with:\n"
            f"  cd {cwd} && {process.args}\n\n"
        )


@pytest.mark.usefixtures("iree_hip_target_type")
class BaseBenchmarkTest(unittest.TestCase):
    directory_created = False
    current_date = datetime.now()
    dir_path_suffix = current_date.strftime("%Y-%m-%d")
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.dirname(cur_dir)
    tests_dir = os.path.dirname(models_dir)
    sharktank_dir = os.path.dirname(tests_dir)
    repo_root = os.path.dirname(sharktank_dir)
    dir_path = Path(repo_root + "/" + dir_path_suffix)

    @classmethod
    def setUpClass(cls):
        """This method will be run once per class to create the directory."""
        if not cls.directory_created:
            if not os.path.exists(cls.dir_path):
                os.makedirs(cls.dir_path)
            cls.directory_created = True

    def setUp(self):
        self.hip_device_id = os.getenv("HIP_DEVICE_ID", default="0")

    def create_file(self, *, suffix, prefix):
        file_path = Path(prefix).with_suffix(suffix)
        f = open(file_path, "w")
        return file_path

    def get_export_cmd(
        self,
        *,
        attention_kernel: str,
        tensor_parallelism_size: int,
        irpa_path: str,
        output_mlir_path: str,
        output_json_path: str,
    ):
        export_args = [
            "python3",
            "-m",
            "sharktank.examples.export_paged_llm_v1",
            "--irpa-file",
            irpa_path,
            "--output-mlir",
            output_mlir_path,
            "--output-config",
            output_json_path,
        ]
        if attention_kernel == "decomposed":
            export_args.append("--attention-kernel")
            export_args.append(attention_kernel)
        elif attention_kernel == "torch_sdpa":
            raise NotImplementedError(
                "attention_kernel torch_sdpa not yet plumbed through"
            )
        if tensor_parallelism_size:
            export_args.append("--tensor-parallelism-size")
            export_args.append(str(tensor_parallelism_size))

        cmd = subprocess.list2cmdline(export_args)
        return cmd

    def get_compile_cmd(
        self, *, output_mlir_path: str, output_vmfb_path: str, args: [str]
    ):
        compile_args = ["iree-compile", output_mlir_path]
        compile_args += args
        compile_args += ["-o", output_vmfb_path]
        cmd = subprocess.list2cmdline(compile_args)
        return cmd

    def export_mlir(
        self,
        *,
        attention_kernel: str,
        tensor_parallelism_size: int,
        irpa_path: str,
        output_mlir_path: str,
        output_json_path: str,
        cwd: str | Path,
    ):
        """Runs export_paged_llm_v1.py and exports an MLIR file.
        Args:
            irpa_path: Path to the model irpa file.
            output_mlir_path: Path to the file to save the exported file.
            output_json_path: Path to the file to save the config json file.
        """
        cmd = self.get_export_cmd(
            attention_kernel=attention_kernel,
            tensor_parallelism_size=tensor_parallelism_size,
            irpa_path=irpa_path,
            output_mlir_path=output_mlir_path,
            output_json_path=output_json_path,
        )
        logging.getLogger().info(f"Launching export command:\n" f"cd {cwd} && {cmd}")
        proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd)
        return_code = proc.returncode
        if return_code != 0:
            raise ExportMlirException(proc, cwd)

    def iree_compile(
        self,
        *,
        mlir_path: str,
        output_vmfb_path: str,
        args: List[str],
        cwd: str | Path,
    ):
        """Compiles an input MLIR file to an output .vmfb file.
        This assumes that the `iree-compile` command is available (usually via PATH).
        Args:
            mlir_path: Path to the input MLIR file.
            output_vmfb_path: Path for the output .vmfb file. The directory must already exist.
            args: List of arguments to pass to `iree-compile`.
            cwd: current working directory
        Raises Exception if compilation fails for some reason.
        """
        cmd = self.get_compile_cmd(
            output_mlir_path=mlir_path,
            output_vmfb_path=output_vmfb_path,
            args=args,
        )
        logging.getLogger().info(f"Launching compile command:\n" f"cd {cwd} && {cmd}")
        proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd)
        return_code = proc.returncode
        if return_code != 0:
            raise IreeCompileException(proc, cwd)

    def iree_benchmark_module(
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
            f"--device=hip://{hip_device_id}",
            "--hip_use_streams=true",
            "--hip_allow_inline_execution=true",
            "--device_allocator=caching",
            f"--module={vmfb_name}",
            f"--parameters=model={irpa_path}",
        ]
        benchmark_args += args
        cmd = subprocess.list2cmdline(benchmark_args)
        logging.getLogger().info(f"Launching run command:\n" f"cd {cwd} && {cmd}")
        proc = subprocess.run(cmd, shell=True, stdout=sys.stdout, cwd=cwd)
        return_code = proc.returncode
        if return_code != 0:
            raise IreeBenchmarkException(proc, cwd, cmd)


class BenchmarkLlama3_1_8B(BaseBenchmarkTest):
    def setUp(self):
        super().setUp()
        # TODO: add numpy files to Azure and download from it
        artifacts_dir = Path("/data/extra/models/llama3.1_8B")
        self.irpa_path = artifacts_dir / "llama8b_f16.irpa"
        self.irpa_path_fp8 = artifacts_dir / "llama8b_fp8.irpa"
        self.tensor_parallelism_size = None
        self.dir_path_8b = self.dir_path / "llama-8b"
        self.temp_dir_8b = Path(self.dir_path_8b)
        self.temp_dir_8b.mkdir(parents=True, exist_ok=True)
        self.iree_compile_args = [
            "--iree-hal-target-backends=rocm",
            f"--iree-hip-target={self.iree_hip_target_type}",
        ]
        self.prefill_args_f16 = artifacts_dir / "prefill_args"
        self.decode_args_f16 = artifacts_dir / "decode_args"
        self.prefill_args_fp8 = artifacts_dir / "prefill_args_fp8"
        self.decode_args_fp8 = artifacts_dir / "decode_args_fp8"
        self.iree_run_prefill_args = [
            "--function=prefill_bs4",
            f"--input=@{self.prefill_args_f16}/tokens.npy",
            f"--input=@{self.prefill_args_f16}/seq_lens.npy",
            f"--input=@{self.prefill_args_f16}/seq_block_ids.npy",
            f"--input=@{self.prefill_args_f16}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_decode_args = [
            "--function=decode_bs4",
            f"--input=@{self.decode_args_f16}/tokens.npy",
            f"--input=@{self.decode_args_f16}/seq_lens.npy",
            f"--input=@{self.decode_args_f16}/start_positions.npy",
            f"--input=@{self.decode_args_f16}/seq_block_ids.npy",
            f"--input=@{self.decode_args_f16}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_prefill_args_fp8 = [
            "--function=prefill_bs4",
            f"--input=@{self.prefill_args_fp8}/tokens.npy",
            f"--input=@{self.prefill_args_fp8}/seq_lens.npy",
            f"--input=@{self.prefill_args_fp8}/seq_block_ids.npy",
            f"--input=@{self.prefill_args_fp8}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_decode_args_fp8 = [
            "--function=decode_bs4",
            f"--input=@{self.decode_args_fp8}/tokens.npy",
            f"--input=@{self.decode_args_fp8}/seq_lens.npy",
            f"--input=@{self.decode_args_fp8}/start_positions.npy",
            f"--input=@{self.decode_args_fp8}/seq_block_ids.npy",
            f"--input=@{self.decode_args_fp8}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]

    @longrun
    @is_mi300x
    def testBenchmark8B_f16_Decomposed(self):
        output_file_name = self.dir_path_8b / "f16_decomposed"
        output_mlir = self.create_file(suffix=".mlir", prefix=output_file_name)
        output_json = self.create_file(suffix=".json", prefix=output_file_name)
        output_vmfb = self.create_file(suffix=".vmfb", prefix=output_file_name)
        self.export_mlir(
            attention_kernel="decomposed",
            tensor_parallelism_size=self.tensor_parallelism_size,
            irpa_path=self.irpa_path,
            output_mlir_path=output_mlir,
            output_json_path=output_json,
            cwd=self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={output_file_name}/files"
        ]
        self.iree_compile(
            mlir_path=output_mlir,
            output_vmfb_path=output_vmfb,
            args=iree_compile_args,
            cwd=self.repo_root,
        )
        # benchmark prefill
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_prefill_args,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_decode_args,
            cwd=self.repo_root,
        )

    @longrun
    @is_mi300x
    @pytest.mark.xfail(reason="torch_sdpa not yet plumbed through", strict=True)
    def testBenchmark8B_f16_Non_Decomposed(self):
        output_file_name = self.dir_path_8b / "f16_torch_sdpa"
        output_mlir = self.create_file(suffix=".mlir", prefix=output_file_name)
        output_json = self.create_file(suffix=".json", prefix=output_file_name)
        output_vmfb = self.create_file(suffix=".vmfb", prefix=output_file_name)
        self.export_mlir(
            attention_kernel="torch_sdpa",
            tensor_parallelism_size=self.tensor_parallelism_size,
            irpa_path=self.irpa_path,
            output_mlir_path=output_mlir,
            output_json_path=output_json,
            cwd=self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={output_file_name}/files"
        ]
        self.iree_compile(
            mlir_path=output_mlir,
            output_vmfb_path=output_vmfb,
            args=iree_compile_args,
            cwd=self.repo_root,
        )
        # benchmark prefill
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_prefill_args,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_decode_args,
            cwd=self.repo_root,
        )

    @longrun
    @is_mi300x
    @pytest.mark.xfail(reason="8B fp8 irpa path not stored yet", strict=True)
    def testBenchmark8B_fp8_Decomposed(self):
        output_file_name = self.dir_path_8b / "fp8_decomposed"
        output_mlir = self.create_file(suffix=".mlir", prefix=output_file_name)
        output_json = self.create_file(suffix=".json", prefix=output_file_name)
        output_vmfb = self.create_file(suffix=".vmfb", prefix=output_file_name)
        self.export_mlir(
            attention_kernel="decomposed",
            tensor_parallelism_size=self.tensor_parallelism_size,
            irpa_path=self.irpa_path_fp8,
            output_mlir_path=output_mlir,
            output_json_path=output_json,
            cwd=self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={output_file_name}/files"
        ]
        self.iree_compile(
            mlir_path=output_mlir,
            output_vmfb_path=output_vmfb,
            args=self.iree_compile_args,
            cwd=self.repo_root,
        )
        # benchmark prefill
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8,
            args=self.iree_run_prefill_args,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8,
            args=self.iree_run_decode_args,
            cwd=self.repo_root,
        )

    @longrun
    @is_mi300x
    @pytest.mark.xfail(reason="torch_sdpa not yet plumbed through", strict=True)
    def testBenchmark8B_fp8_Non_Decomposed(self):
        output_file_name = self.dir_path_8b / "fp8_torch_sdpa"
        output_mlir = self.create_file(suffix=".mlir", prefix=output_file_name)
        output_json = self.create_file(suffix=".json", prefix=output_file_name)
        output_vmfb = self.create_file(suffix=".vmfb", prefix=output_file_name)
        self.export_mlir(
            attention_kernel="torch_sdpa",
            tensor_parallelism_size=self.tensor_parallelism_size,
            irpa_path=self.irpa_path_fp8,
            output_mlir_path=output_mlir,
            output_json_path=output_json,
            cwd=self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={output_file_name}/files"
        ]
        self.iree_compile(
            mlir_path=output_mlir,
            output_vmfb_path=output_vmfb,
            args=self.iree_compile_args,
            cwd=self.repo_root,
        )
        # benchmark prefill
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8,
            args=self.iree_run_prefill_args_fp8,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8,
            args=self.iree_run_decode_args_fp8,
            cwd=self.repo_root,
        )


class BenchmarkLlama3_1_70B(BaseBenchmarkTest):
    def setUp(self):
        super().setUp()
        # TODO: add numpy files to Azure and download from it
        artifacts_dir = Path("/data/extra/models/llama3.1_70B")
        self.irpa_path = artifacts_dir / "llama70b_f16.irpa"
        self.irpa_path_fp8 = artifacts_dir / "llama70b_fp8.irpa"
        self.tensor_parallelism_size = 1
        self.dir_path_70b = self.dir_path / "llama-70b"
        self.temp_dir_70b = Path(self.dir_path_70b)
        self.temp_dir_70b.mkdir(parents=True, exist_ok=True)
        self.iree_compile_args = [
            "--iree-hal-target-backends=rocm",
            f"--iree-hip-target={self.iree_hip_target_type}",
        ]
        self.prefill_args_f16 = artifacts_dir / "prefill_args"
        self.decode_args_f16 = artifacts_dir / "decode_args"
        self.prefill_args_fp8 = artifacts_dir / "prefill_args_fp8"
        self.decode_args_fp8 = artifacts_dir / "decode_args_fp8"
        self.iree_run_prefill_args = [
            "--function=prefill_bs4",
            f"--input=@{self.prefill_args_f16}/tokens.npy",
            f"--input=@{self.prefill_args_f16}/seq_lens.npy",
            f"--input=@{self.prefill_args_f16}/seq_block_ids.npy",
            f"--input=@{self.prefill_args_f16}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_decode_args = [
            "--function=decode_bs4",
            f"--input=@{self.decode_args_f16}/tokens.npy",
            f"--input=@{self.decode_args_f16}/seq_lens.npy",
            f"--input=@{self.decode_args_f16}/start_positions.npy",
            f"--input=@{self.decode_args_f16}/seq_block_ids.npy",
            f"--input=@{self.decode_args_f16}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_prefill_args_fp8 = [
            "--function=prefill_bs4",
            f"--input=@{self.prefill_args_fp8}/tokens.npy",
            f"--input=@{self.prefill_args_fp8}/seq_lens.npy",
            f"--input=@{self.prefill_args_fp8}/seq_block_ids.npy",
            f"--input=@{self.prefill_args_fp8}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_decode_args_fp8 = [
            "--function=decode_bs4",
            f"--input=@{self.decode_args_fp8}/tokens.npy",
            f"--input=@{self.decode_args_fp8}/seq_lens.npy",
            f"--input=@{self.decode_args_fp8}/start_positions.npy",
            f"--input=@{self.decode_args_fp8}/seq_block_ids.npy",
            f"--input=@{self.decode_args_fp8}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]

    @longrun
    @is_mi300x
    @pytest.mark.xfail(reason="70b f16 irpa path not stored yet", strict=True)
    def testBenchmark70B_f16_Decomposed(self):
        output_file_name = self.dir_path_70b / "f16_decomposed"
        output_mlir = self.create_file(suffix=".mlir", prefix=output_file_name)
        output_json = self.create_file(suffix=".json", prefix=output_file_name)
        output_vmfb = self.create_file(suffix=".vmfb", prefix=output_file_name)
        self.export_mlir(
            attention_kernel="decomposed",
            tensor_parallelism_size=self.tensor_parallelism_size,
            irpa_path=self.irpa_path,
            output_mlir_path=output_mlir,
            output_json_path=output_json,
            cwd=self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={output_file_name}/files"
        ]
        self.iree_compile(
            mlir_path=output_mlir,
            output_vmfb_path=output_vmfb,
            args=iree_compile_args,
            cwd=self.repo_root,
        )
        # benchmark prefill
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_prefill_args,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_decode_args,
            cwd=self.repo_root,
        )

    @longrun
    @is_mi300x
    @pytest.mark.xfail(reason="torch_sdpa not yet plumbed through", strict=True)
    def testBenchmark70B_f16_Non_Decomposed(self):
        output_file_name = self.dir_path_70b / "f16_torch_sdpa"
        output_mlir = self.create_file(suffix=".mlir", prefix=output_file_name)
        output_json = self.create_file(suffix=".json", prefix=output_file_name)
        output_vmfb = self.create_file(suffix=".vmfb", prefix=output_file_name)
        self.export_mlir(
            attention_kernel="torch_sdpa",
            tensor_parallelism_size=self.tensor_parallelism_size,
            irpa_path=self.irpa_path,
            output_mlir_path=output_mlir,
            output_json_path=output_json,
            cwd=self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={output_file_name}/files"
        ]
        self.iree_compile(
            mlir_path=output_mlir,
            output_vmfb_path=output_vmfb,
            args=iree_compile_args,
            cwd=self.repo_root,
        )
        # benchmark prefill
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_prefill_args,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_decode_args,
            cwd=self.repo_root,
        )

    @longrun
    @is_mi300x
    @pytest.mark.xfail(reason="70B fp8 irpa path not stored yet", strict=True)
    def testBenchmark70B_fp8_Decomposed(self):
        output_file_name = self.dir_path_70b / "fp8_decomposed"
        output_mlir = self.create_file(suffix=".mlir", prefix=output_file_name)
        output_json = self.create_file(suffix=".json", prefix=output_file_name)
        output_vmfb = self.create_file(suffix=".vmfb", prefix=output_file_name)
        self.export_mlir(
            attention_kernel="decomposed",
            tensor_parallelism_size=self.tensor_parallelism_size,
            irpa_path=self.irpa_path_fp8,
            output_mlir_path=output_mlir,
            output_json_path=output_json,
            cwd=self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={output_file_name}/files"
        ]
        self.iree_compile(
            mlir_path=output_mlir,
            output_vmfb_path=output_vmfb,
            args=self.iree_compile_args,
            cwd=self.repo_root,
        )
        # benchmark prefill
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8,
            args=self.iree_run_prefill_args,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8,
            args=self.iree_run_decode_args,
            cwd=self.repo_root,
        )

    @longrun
    @is_mi300x
    @pytest.mark.xfail(reason="torch_sdpa not yet plumbed through", strict=True)
    def testBenchmark70B_fp8_Non_Decomposed(self):
        output_file_name = self.dir_path_70b / "fp8_torch_sdpa"
        output_mlir = self.create_file(suffix=".mlir", prefix=output_file_name)
        output_json = self.create_file(suffix=".json", prefix=output_file_name)
        output_vmfb = self.create_file(suffix=".vmfb", prefix=output_file_name)
        self.export_mlir(
            attention_kernel="torch_sdpa",
            tensor_parallelism_size=self.tensor_parallelism_size,
            irpa_path=self.irpa_path_fp8,
            output_mlir_path=output_mlir,
            output_json_path=output_json,
            cwd=self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={output_file_name}/files"
        ]
        self.iree_compile(
            mlir_path=output_mlir,
            output_vmfb_path=output_vmfb,
            args=self.iree_compile_args,
            cwd=self.repo_root,
        )
        # benchmark prefill
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8,
            args=self.iree_run_prefill_args_fp8,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8,
            args=self.iree_run_decode_args_fp8,
            cwd=self.repo_root,
        )


class BenchmarkLlama3_1_405B(BaseBenchmarkTest):
    def setUp(self):
        super().setUp()
        # TODO: add numpy files to Azure and download from it
        artifacts_dir = Path("/data/extra/models/llama3.1_405B")
        self.irpa_path = artifacts_dir / "llama405b_f16.irpa"
        self.irpa_path_fp8 = artifacts_dir / "llama405b_fp8.irpa"
        self.tensor_parallelism_size = 8
        self.dir_path_405b = self.dir_path / "llama-405b"
        self.temp_dir_405b = Path(self.dir_path_405b)
        self.temp_dir_405b.mkdir(parents=True, exist_ok=True)
        self.iree_compile_args = [
            "--iree-hal-target-backends=rocm",
            f"--iree-hip-target={self.iree_hip_target_type}",
        ]
        self.prefill_args_f16 = artifacts_dir / "prefill_args"
        self.decode_args_f16 = artifacts_dir / "decode_args"
        self.prefill_args_fp8 = artifacts_dir / "prefill_args_fp8"
        self.decode_args_fp8 = artifacts_dir / "decode_args_fp8"
        self.iree_run_prefill_args = [
            "--function=prefill_bs4",
            f"--input=@{self.prefill_args_f16}/tokens.npy",
            f"--input=@{self.prefill_args_f16}/seq_lens.npy",
            f"--input=@{self.prefill_args_f16}/seq_block_ids.npy",
            f"--input=@{self.prefill_args_f16}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_decode_args = [
            "--function=decode_bs4",
            f"--input=@{self.decode_args_f16}/tokens.npy",
            f"--input=@{self.decode_args_f16}/seq_lens.npy",
            f"--input=@{self.decode_args_f16}/start_positions.npy",
            f"--input=@{self.decode_args_f16}/seq_block_ids.npy",
            f"--input=@{self.decode_args_f16}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_prefill_args_fp8 = [
            "--function=prefill_bs4",
            f"--input=@{self.prefill_args_fp8}/tokens.npy",
            f"--input=@{self.prefill_args_fp8}/seq_lens.npy",
            f"--input=@{self.prefill_args_fp8}/seq_block_ids.npy",
            f"--input=@{self.prefill_args_fp8}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_decode_args_fp8 = [
            "--function=decode_bs4",
            f"--input=@{self.decode_args_fp8}/tokens.npy",
            f"--input=@{self.decode_args_fp8}/seq_lens.npy",
            f"--input=@{self.decode_args_fp8}/start_positions.npy",
            f"--input=@{self.decode_args_fp8}/seq_block_ids.npy",
            f"--input=@{self.decode_args_fp8}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]

    @longrun
    @is_mi300x
    @pytest.mark.xfail(reason="405B f16 irpa path not stored yet", strict=True)
    def testBenchmark405B_f16_Decomposed(self):
        output_file_name = self.dir_path_405b / "f16_decomposed"
        output_mlir = self.create_file(suffix=".mlir", prefix=output_file_name)
        output_json = self.create_file(suffix=".json", prefix=output_file_name)
        output_vmfb = self.create_file(suffix=".vmfb", prefix=output_file_name)
        self.export_mlir(
            attention_kernel="decomposed",
            tensor_parallelism_size=self.tensor_parallelism_size,
            irpa_path=self.irpa_path,
            output_mlir_path=output_mlir,
            output_json_path=output_json,
            cwd=self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={output_file_name}/files"
        ]
        self.iree_compile(
            mlir_path=output_mlir,
            output_vmfb_path=output_vmfb,
            args=iree_compile_args,
            cwd=self.repo_root,
        )
        # benchmark prefill
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_prefill_args,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_decode_args,
            cwd=self.repo_root,
        )

    @longrun
    @is_mi300x
    @pytest.mark.xfail(reason="torch_sdpa not yet plumbed through", strict=True)
    def testBenchmark405B_f16_Non_Decomposed(self):
        output_file_name = self.dir_path_405b / "f16_torch_sdpa"
        output_mlir = self.create_file(suffix=".mlir", prefix=output_file_name)
        output_json = self.create_file(suffix=".json", prefix=output_file_name)
        output_vmfb = self.create_file(suffix=".vmfb", prefix=output_file_name)
        self.export_mlir(
            attention_kernel="torch_sdpa",
            tensor_parallelism_size=self.tensor_parallelism_size,
            irpa_path=self.irpa_path,
            output_mlir_path=output_mlir,
            output_json_path=output_json,
            cwd=self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={output_file_name}/files"
        ]
        self.iree_compile(
            mlir_path=output_mlir,
            output_vmfb_path=output_vmfb,
            args=iree_compile_args,
            cwd=self.repo_root,
        )
        # benchmark prefill
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_prefill_args,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_decode_args,
            cwd=self.repo_root,
        )

    @longrun
    @is_mi300x
    @pytest.mark.xfail(reason="405B fp8 irpa path not stored yet", strict=True)
    def testBenchmark405B_fp8_Decomposed(self):
        output_file_name = self.dir_path_405b / "fp8_decomposed"
        output_mlir = self.create_file(suffix=".mlir", prefix=output_file_name)
        output_json = self.create_file(suffix=".json", prefix=output_file_name)
        output_vmfb = self.create_file(suffix=".vmfb", prefix=output_file_name)
        self.export_mlir(
            attention_kernel="decomposed",
            tensor_parallelism_size=self.tensor_parallelism_size,
            irpa_path=self.irpa_path_fp8,
            output_mlir_path=output_mlir,
            output_json_path=output_json,
            cwd=self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={output_file_name}/files"
        ]
        self.iree_compile(
            mlir_path=output_mlir,
            output_vmfb_path=output_vmfb,
            args=self.iree_compile_args,
            cwd=self.repo_root,
        )
        # benchmark prefill
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_prefill_args,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_decode_args,
            cwd=self.repo_root,
        )

    @longrun
    @is_mi300x
    @pytest.mark.xfail(reason="torch_sdpa not yet plumbed through", strict=True)
    def testBenchmark405B_fp8_Non_Decomposed(self):
        output_file_name = self.dir_path_405b / "fp8_torch_sdpa"
        output_mlir = self.create_file(suffix=".mlir", prefix=output_file_name)
        output_json = self.create_file(suffix=".json", prefix=output_file_name)
        output_vmfb = self.create_file(suffix=".vmfb", prefix=output_file_name)
        self.export_mlir(
            attention_kernel="torch_sdpa",
            tensor_parallelism_size=self.tensor_parallelism_size,
            irpa_path=self.irpa_path_fp8,
            output_mlir_path=output_mlir,
            output_json_path=output_json,
            cwd=self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={output_file_name}/files"
        ]
        self.iree_compile(
            mlir_path=output_mlir,
            output_vmfb_path=output_vmfb,
            args=self.iree_compile_args,
            cwd=self.repo_root,
        )
        # benchmark prefill
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8,
            args=self.iree_run_prefill_args_fp8,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8,
            args=self.iree_run_decode_args_fp8,
            cwd=self.repo_root,
        )


if __name__ == "__main__":
    unittest.main()
