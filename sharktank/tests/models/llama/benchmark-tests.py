# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import os
import unittest
import subprocess
from pathlib import Path
from typing import List


class BaseBenchmarkTest(unittest.TestCase):
    def setUp(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_export_cmd(self, irpa_path: str, output_mlir_path: str, output_json_path: str):
        export_args = [
            "python3",
            "-m",
            "sharktank.examples.export_paged_llm_v1",
            "--irpa-file",
            irpa_path,
            "--output-mlir",
            output_mlir_path,
            "--output-config",
            output_json_path
        ]
        cmd = subprocess.list2cmdline(export_args)
        return cmd

    def get_compile_cmd(self, mlir_path: str, output_file: str, args: [str]):
        compile_args = [
            "iree-compile",
            mlir_path
        ]
        compile_args += args
        compile_args += ["-o", output_file]
        cmd = subprocess.list2cmdline(compile_args)
        return cmd
    
    def export_mlir(self, irpa_path: str, output_mlir_path: str, output_json_path: str, cwd: str | Path):
        """Runs export_paged_llm_v1.py and exports an MLIR file.
        Args:
            irpa_path: Path to the model irpa file.
            output_mlir_path: Path to the file to save the exported file.
            output_json_path: Path to the file to save the config json file.
        """
        cmd = self.get_export_cmd(irpa_path, output_mlir_path, output_json_path)
        logging.getLogger().info(
            f"Launching export command:\n"
            f"cd {cwd} && {cmd}"
        )
        proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd)
        return_code = proc.returncode
        if return_code != 0:
            raise Exception(f"{cmd} failed to export.")
    
    def iree_compile(self, mlir_path: str, output_file: str, args: List[str], cwd: str | Path):
        """Compiles an input MLIR file to an output .vmfb file.
        This assumes that the `iree-compile` command is available (usually via PATH).
        Args:
            mlir_path: Path to the input MLIR file.
            output_file: Path for the output file. The directory must already exist.
            args: List of arguments to pass to `iree-compile`.
            cwd: current working directory
        Raises Exception if compilation fails for some reason.
        """
        cmd = self.get_compile_cmd(mlir_path, output_file, args)
        logging.getLogger().info(
            f"Launching compile command:\n"
            f"cd {cwd} && {cmd}"
        )
        proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd)
        return_code = proc.returncode
        if return_code != 0:
            raise Exception(f"{cmd} failed to compile.")

    def iree_benchmark_module(self, hip_device_id: str, vmfb_name: str, irpa_path: str, args: List[str], cwd: str | Path):
        """Runs a compiled program with the given args using `iree-benchmark-module`.
        This assumes that the `iree-benchmark-module` command is available (usually via PATH).
        Args:
            vmfb_name: Name of the .vmfb file (relative to `cwd`).
            args: List of arguments to pass to `iree-benchmark-module`.
            cwd: Working directory to run the command within. (either string or Path works)
            compile_cmd: Command used to compile the program, for inclusion in error messages.
        Raises Exception if running fails for some reason.
        """
        run_args = [
            f"ROCR_VISIBLE_DEVICES={hip_device_id}",
            "iree-benchmark-module",
            f"--device=hip://{hip_device_id}",
            "--hip_use_streams=true",
            "--hip_allow_inline_execution=true",
            "--device_allocator=caching",
            f"--module={vmfb_name}",
            f"--parameters=model={irpa_path}"
        ]
        run_args += args
        cmd = subprocess.list2cmdline(run_args)
        logging.getLogger().info(
            f"Launching run command:\n"
            f"cd {cwd} && {cmd}"
        )
        proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd)
        return_code = proc.returncode
        if return_code != 0:
            raise Exception(f"{cmd} failed to run")


class BenchmarkLlama3_1_8B_f16(BaseBenchmarkTest):
    def setUp(self):
        # TODO: add numpy files to Azure and download from it
        self.repo_root = "/home/avsharma/SHARK-Platform/"#os.getenv("SHARK_PLATFORM_REPO_ROOT")
        artifacts_dir = "/data/extra/models/llama3.1_8B/"
        self.irpa_path = artifacts_dir + "llama8b_f16.irpa"
        self.output_mlir = self.repo_root + "llama8b_f16.mlir"
        self.output_json = self.repo_root + "llama8b_f16.json"
        self.output_vmfb = self.repo_root + "llama8b_f16.vmfb"
        self.iree_compile_args = ["--iree-hal-target-backends=rocm", "--iree-hip-target=gfx942", f"--iree-hal-dump-executable-files-to={self.repo_root}/files/llama"]
        self.prefill_args_f16 = artifacts_dir + "prefill_args"
        self.decode_args_f16 = artifacts_dir + "decode_args"
        self.iree_run_prefill_args = ["--function=prefill_bs4", f"--input=@{self.prefill_args_f16}/tokens.npy", f"--input=@{self.prefill_args_f16}/seq_lens.npy", f"--input=@{self.prefill_args_f16}/seq_block_ids.npy", f"--input=@{self.prefill_args_f16}/cache_state_f16.npy", "--benchmark_repetitions=3"]
        self.iree_run_decode_args = ["--function=decode_bs4", f"--input=@{self.decode_args_f16}/tokens.npy", f"--input=@{self.decode_args_f16}/seq_lens.npy", f"--input=@{self.decode_args_f16}/start_positions.npy", f"--input=@{self.decode_args_f16}/seq_block_ids.npy", f"--input=@{self.decode_args_f16}/cache_state_f16.npy", "--benchmark_repetitions=3"]

    def testExport8B_f16(self):
        self.export_mlir(self.irpa_path, self.output_mlir, self.output_json, self.repo_root)
        self.iree_compile(self.output_mlir, self.output_vmfb, self.iree_compile_args, self.repo_root)
        self.iree_benchmark_module("0", self.output_vmfb, self.irpa_path, self.iree_run_prefill_args, self.repo_root)
