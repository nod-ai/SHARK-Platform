# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import os
import sys
import unittest
import pytest
import subprocess
from pathlib import Path
from typing import List

longrun = pytest.mark.skipif("not config.getoption('longrun')")


class BaseBenchmarkTest(unittest.TestCase):
    def setUp(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_export_cmd(
        self,
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
        if attention_kernel in ["decomposed", "torch_sdpa"]:
            export_args.append("--attention-kernel")
            export_args.append(attention_kernel)
        if tensor_parallelism_size:
            export_args.append("--tensor-parallelism-size")
            export_args.append(str(tensor_parallelism_size))

        cmd = subprocess.list2cmdline(export_args)
        return cmd

    def get_compile_cmd(self, mlir_path: str, output_file: str, args: [str]):
        compile_args = ["iree-compile", mlir_path]
        compile_args += args
        compile_args += ["-o", output_file]
        cmd = subprocess.list2cmdline(compile_args)
        return cmd

    def export_mlir(
        self,
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
            attention_kernel,
            tensor_parallelism_size,
            irpa_path,
            output_mlir_path,
            output_json_path,
        )
        logging.getLogger().info(f"Launching export command:\n" f"cd {cwd} && {cmd}")
        proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd)
        return_code = proc.returncode
        if return_code != 0:
            raise Exception(f"{cmd} failed to export.")

    def iree_compile(
        self, mlir_path: str, output_file: str, args: List[str], cwd: str | Path
    ):
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
        logging.getLogger().info(f"Launching compile command:\n" f"cd {cwd} && {cmd}")
        proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd)
        return_code = proc.returncode
        if return_code != 0:
            raise Exception(f"{cmd} failed to compile.")

    def iree_benchmark_module(
        self,
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
            raise Exception(f"{cmd} failed to run")

    def cleanup_output_files(
        self, output_mlir_path: str, output_json_path: str, output_file: str
    ):
        try:
            # Removing output_mlir_path
            subprocess.run(f"rm {output_mlir_path}", shell=True, check=True)
            print(f"Removed: {output_mlir_path}")

            # Removing output_json_path
            subprocess.run(f"rm {output_json_path}", shell=True, check=True)
            print(f"Removed: {output_json_path}")

            # Removing output_file
            subprocess.run(f"rm {output_file}", shell=True, check=True)
            print(f"Removed: {output_file}")

        except subprocess.CalledProcessError as e:
            print(f"Error removing files: {e}")


class BenchmarkLlama3_1_8B(BaseBenchmarkTest):
    def setUp(self):
        # TODO: add numpy files to Azure and download from it
        self.repo_root = os.getenv("SHARK_PLATFORM_REPO_ROOT")
        artifacts_dir = "/data/extra/models/llama3.1_8B/"
        self.irpa_path = artifacts_dir + "llama8b_f16.irpa"
        self.irpa_path_fp8 = artifacts_dir + "llama8b_fp8.irpa"
        self.tensor_parallelism_size = None
        self.iree_compile_args = [
            "--iree-hal-target-backends=rocm",
            "--iree-hip-target=gfx942",
        ]
        self.prefill_args_f16 = artifacts_dir + "prefill_args"
        self.decode_args_f16 = artifacts_dir + "decode_args"
        self.prefill_args_fp8 = artifacts_dir + "prefill_args_fp8"
        self.decode_args_fp8 = artifacts_dir + "decode_args_fp8"
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
    def testBenchmark8B_f16_Decomposed(self):
        output_mlir = self.repo_root + "llama8b_f16_decomposed.mlir"
        output_json = self.repo_root + "llama8b_f16_decomposed.json"
        output_vmfb = self.repo_root + "llama8b_f16_decomposed.vmfb"
        self.export_mlir(
            "decomposed",
            self.tensor_parallelism_size,
            self.irpa_path,
            output_mlir,
            output_json,
            self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={self.repo_root}/files/llama-8b/f16-decomposed"
        ]
        self.iree_compile(output_mlir, output_vmfb, iree_compile_args, self.repo_root)
        # benchmark prefill
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path,
            self.iree_run_prefill_args,
            self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path,
            self.iree_run_decode_args,
            self.repo_root,
        )
        self.cleanup_output_files(output_mlir, output_json, output_vmfb)

    @pytest.mark.skip(reason="TODO: Need to plumb through attention_kernel")
    @longrun
    def testBenchmark8B_f16_Non_Decomposed(self):
        output_mlir = self.repo_root + "llama8b_f16_torch_sdpa.mlir"
        output_json = self.repo_root + "llama8b_f16_torch_sdpa.json"
        output_vmfb = self.repo_root + "llama8b_f16_torch_sdpa.vmfb"
        self.export_mlir(
            "torch_sdpa",
            self.tensor_parallelism_size,
            self.irpa_path,
            output_mlir,
            output_json,
            self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={self.repo_root}/files/llama-8b/f16-torch-sdpa"
        ]
        self.iree_compile(output_mlir, output_vmfb, iree_compile_args, self.repo_root)
        # benchmark prefill
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path,
            self.iree_run_prefill_args,
            self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path,
            self.iree_run_decode_args,
            self.repo_root,
        )
        self.cleanup_output_files(output_mlir, output_json, output_vmfb)

    @pytest.mark.xfail
    @longrun
    def testBenchmark8B_fp8_Decomposed(self):
        output_mlir = self.repo_root + "llama8b_fp8_decomposed.mlir"
        output_json = self.repo_root + "llama8b_fp8_decomposed.json"
        output_vmfb = self.repo_root + "llama8b_fp8_decomposed.vmfb"
        self.export_mlir(
            "decomposed",
            self.tensor_parallelism_size,
            self.irpa_path_fp8,
            output_mlir,
            output_json,
            self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={self.repo_root}/files/llama-8b/fp8-decomposed"
        ]
        self.iree_compile(
            output_mlir,
            output_vmfb,
            self.iree_compile_args,
            self.repo_root,
        )
        # benchmark prefill
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path,
            self.iree_run_prefill_args,
            self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path,
            self.iree_run_decode_args,
            self.repo_root,
        )
        self.cleanup_output_files(output_mlir, output_json, output_vmfb)

    @pytest.mark.xfail
    @longrun
    def testBenchmark8B_fp8_Non_Decomposed(self):
        output_mlir = self.repo_root + "llama8b_fp8_torch_sdpa.mlir"
        output_json = self.repo_root + "llama8b_fp8_torch_sdpa.json"
        output_vmfb = self.repo_root + "llama8b_fp8_torch_sdpa.vmfb"
        self.export_mlir(
            "torch_sdpa",
            self.tensor_parallelism_size,
            self.irpa_path_fp8,
            output_mlir,
            output_json,
            self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={self.repo_root}/files/llama-8b/fp8-torch-sdpa"
        ]
        self.iree_compile(
            output_mlir,
            output_vmfb,
            self.iree_compile_args,
            self.repo_root,
        )
        # benchmark prefill
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path_fp8,
            self.iree_run_prefill_args_fp8,
            self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path_fp8,
            self.iree_run_decode_args_fp8,
            self.repo_root,
        )
        self.cleanup_output_files(output_mlir, output_json, output_vmfb)


class BenchmarkLlama3_1_70B(BaseBenchmarkTest):
    def setUp(self):
        # TODO: add numpy files to Azure and download from it
        self.repo_root = os.getenv("SHARK_PLATFORM_REPO_ROOT")
        artifacts_dir = "/data/extra/models/llama3.1_70B/"
        self.irpa_path = artifacts_dir + "llama70b_f16.irpa"
        self.irpa_path_fp8 = artifacts_dir + "llama70b_fp8.irpa"
        self.tensor_parallelism_size = 1
        self.iree_compile_args = [
            "--iree-hal-target-backends=rocm",
            "--iree-hip-target=gfx942",
        ]
        self.prefill_args_f16 = artifacts_dir + "prefill_args"
        self.decode_args_f16 = artifacts_dir + "decode_args"
        self.prefill_args_fp8 = artifacts_dir + "prefill_args_fp8"
        self.decode_args_fp8 = artifacts_dir + "decode_args_fp8"
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

    @pytest.mark.xfail
    @longrun
    def testBenchmark70B_f16_Decomposed(self):
        output_mlir = self.repo_root + "llama70b_f16_decomposed.mlir"
        output_json = self.repo_root + "llama70b_f16_decomposed.json"
        output_vmfb = self.repo_root + "llama70b_f16_decomposed.vmfb"
        self.export_mlir(
            "decomposed",
            self.tensor_parallelism_size,
            self.irpa_path,
            output_mlir,
            output_json,
            self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={self.repo_root}/files/llama-70b/f16-decomposed"
        ]
        self.iree_compile(output_mlir, output_vmfb, iree_compile_args, self.repo_root)
        # benchmark prefill
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path,
            self.iree_run_prefill_args,
            self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path,
            self.iree_run_decode_args,
            self.repo_root,
        )
        self.cleanup_output_files(output_mlir, output_json, output_vmfb)

    @pytest.mark.skip(reason="TODO: Need to plumb through attention_kernel")
    @longrun
    def testBenchmark70B_f16_Non_Decomposed(self):
        output_mlir = self.repo_root + "llama70b_f16_torch_sdpa.mlir"
        output_json = self.repo_root + "llama70b_f16_torch_sdpa.json"
        output_vmfb = self.repo_root + "llama70b_f16_torch_sdpa.vmfb"
        self.export_mlir(
            "torch_sdpa",
            self.tensor_parallelism_size,
            self.irpa_path,
            output_mlir,
            output_json,
            self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={self.repo_root}/files/llama-70b/f16-torch-sdpa"
        ]
        self.iree_compile(output_mlir, output_vmfb, iree_compile_args, self.repo_root)
        # benchmark prefill
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path,
            self.iree_run_prefill_args,
            self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path,
            self.iree_run_decode_args,
            self.repo_root,
        )
        self.cleanup_output_files(output_mlir, output_json, output_vmfb)

    @pytest.mark.xfail
    @longrun
    def testBenchmark70B_fp8_Decomposed(self):
        output_mlir = self.repo_root + "llama70b_fp8_decomposed.mlir"
        output_json = self.repo_root + "llama70b_fp8_decomposed.json"
        output_vmfb = self.repo_root + "llama70b_fp8_decomposed.vmfb"
        self.export_mlir(
            "decomposed",
            self.tensor_parallelism_size,
            self.irpa_path_fp8,
            output_mlir,
            output_json,
            self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={self.repo_root}/files/llama-70b/fp8-decomposed"
        ]
        self.iree_compile(
            output_mlir,
            output_vmfb,
            self.iree_compile_args,
            self.repo_root,
        )
        # benchmark prefill
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path,
            self.iree_run_prefill_args,
            self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path,
            self.iree_run_decode_args,
            self.repo_root,
        )
        self.cleanup_output_files(output_mlir, output_json, output_vmfb)

    @pytest.mark.xfail
    @longrun
    def testBenchmark70B_fp8_Non_Decomposed(self):
        output_mlir = self.repo_root + "llama70b_fp8_torch_sdpa.mlir"
        output_json = self.repo_root + "llama70b_fp8_torch_sdpa.json"
        output_vmfb = self.repo_root + "llama70b_fp8_torch_sdpa.vmfb"
        self.export_mlir(
            "torch_sdpa",
            self.tensor_parallelism_size,
            self.irpa_path_fp8,
            output_mlir,
            output_json,
            self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={self.repo_root}/files/llama-70b/fp8-torch-sdpa"
        ]
        self.iree_compile(
            output_mlir,
            output_vmfb,
            self.iree_compile_args,
            self.repo_root,
        )
        # benchmark prefill
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path_fp8,
            self.iree_run_prefill_args_fp8,
            self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path_fp8,
            self.iree_run_decode_args_fp8,
            self.repo_root,
        )
        self.cleanup_output_files(output_mlir, output_json, output_vmfb)


class BenchmarkLlama3_1_405B(BaseBenchmarkTest):
    def setUp(self):
        # TODO: add numpy files to Azure and download from it
        self.repo_root = os.getenv("SHARK_PLATFORM_REPO_ROOT")
        artifacts_dir = "/data/extra/models/llama3.1_405B/"
        self.irpa_path = artifacts_dir + "llama405b_f16.irpa"
        self.irpa_path_fp8 = artifacts_dir + "llama405b_fp8.irpa"
        self.tensor_parallelism_size = 8
        self.iree_compile_args = [
            "--iree-hal-target-backends=rocm",
            "--iree-hip-target=gfx942",
        ]
        self.prefill_args_f16 = artifacts_dir + "prefill_args"
        self.decode_args_f16 = artifacts_dir + "decode_args"
        self.prefill_args_fp8 = artifacts_dir + "prefill_args_fp8"
        self.decode_args_fp8 = artifacts_dir + "decode_args_fp8"
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

    @pytest.mark.xfail
    @longrun
    def testBenchmark405B_f16_Decomposed(self):
        output_mlir = self.repo_root + "llama405b_f16_decomposed.mlir"
        output_json = self.repo_root + "llama405b_f16_decomposed.json"
        output_vmfb = self.repo_root + "llama405b_f16_decomposed.vmfb"
        self.export_mlir(
            "decomposed",
            self.tensor_parallelism_size,
            self.irpa_path,
            output_mlir,
            output_json,
            self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={self.repo_root}/files/llama-405b/f16-decomposed"
        ]
        self.iree_compile(output_mlir, output_vmfb, iree_compile_args, self.repo_root)
        # benchmark prefill
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path,
            self.iree_run_prefill_args,
            self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path,
            self.iree_run_decode_args,
            self.repo_root,
        )
        self.cleanup_output_files(output_mlir, output_json, output_vmfb)

    @pytest.mark.skip(reason="TODO: Need to plumb through attention_kernel")
    @longrun
    def testBenchmark405B_f16_Non_Decomposed(self):
        output_mlir = self.repo_root + "llama405b_f16_torch_sdpa.mlir"
        output_json = self.repo_root + "llama405b_f16_torch_sdpa.json"
        output_vmfb = self.repo_root + "llama405b_f16_torch_sdpa.vmfb"
        self.export_mlir(
            "torch_sdpa",
            self.tensor_parallelism_size,
            self.irpa_path,
            output_mlir,
            output_json,
            self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={self.repo_root}/files/llama-405b/f16-torch-sdpa"
        ]
        self.iree_compile(output_mlir, output_vmfb, iree_compile_args, self.repo_root)
        # benchmark prefill
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path,
            self.iree_run_prefill_args,
            self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path,
            self.iree_run_decode_args,
            self.repo_root,
        )
        self.cleanup_output_files(output_mlir, output_json, output_vmfb)

    @pytest.mark.xfail
    @longrun
    def testBenchmark405B_fp8_Decomposed(self):
        output_mlir = self.repo_root + "llama405b_fp8_decomposed.mlir"
        output_json = self.repo_root + "llama405b_fp8_decomposed.json"
        output_vmfb = self.repo_root + "llama405b_fp8_decomposed.vmfb"
        self.export_mlir(
            "decomposed",
            self.tensor_parallelism_size,
            self.irpa_path_fp8,
            output_mlir,
            output_json,
            self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={self.repo_root}/files/llama-405b/fp8-decomposed"
        ]
        self.iree_compile(
            output_mlir,
            output_vmfb,
            self.iree_compile_args,
            self.repo_root,
        )
        # benchmark prefill
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path,
            self.iree_run_prefill_args,
            self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path,
            self.iree_run_decode_args,
            self.repo_root,
        )
        self.cleanup_output_files(output_mlir, output_json, output_vmfb)

    @pytest.mark.xfail
    @longrun
    def testBenchmark405B_fp8_Non_Decomposed(self):
        output_mlir = self.repo_root + "llama405b_fp8_torch_sdpa.mlir"
        output_json = self.repo_root + "llama405b_fp8_torch_sdpa.json"
        output_vmfb = self.repo_root + "llama405b_fp8_torch_sdpa.vmfb"
        self.export_mlir(
            "torch_sdpa",
            self.tensor_parallelism_size,
            self.irpa_path_fp8,
            output_mlir,
            output_json,
            self.repo_root,
        )
        iree_compile_args = self.iree_compile_args + [
            f"--iree-hal-dump-executable-files-to={self.repo_root}/files/llama-405b/fp8-torch-sdpa"
        ]
        self.iree_compile(
            output_mlir,
            output_vmfb,
            self.iree_compile_args,
            self.repo_root,
        )
        # benchmark prefill
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path_fp8,
            self.iree_run_prefill_args_fp8,
            self.repo_root,
        )
        # benchmark decode
        self.iree_benchmark_module(
            "0",
            output_vmfb,
            self.irpa_path_fp8,
            self.iree_run_decode_args_fp8,
            self.repo_root,
        )
        self.cleanup_output_files(output_mlir, output_json, output_vmfb)


if __name__ == "__main__":
    unittest.main()
