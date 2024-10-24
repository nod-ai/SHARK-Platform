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
from sharktank.utils.export_artifacts import ExportArtifacts

longrun = pytest.mark.skipif("not config.getoption('longrun')")
is_mi300x = pytest.mark.skipif("config.getoption('iree_hip_target') != 'gfx942'")


@pytest.mark.usefixtures("get_iree_flags")
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


class BenchmarkLlama3_1_8B(BaseBenchmarkTest):
    def setUp(self):
        super().setUp()
        # TODO: add numpy files to Azure and download from it
        self.artifacts_dir = Path("/data/llama-3.1/8b")
        self.irpa_path = self.artifacts_dir / "llama8b_f16.irpa"
        self.irpa_path_fp8 = self.artifacts_dir / "llama8b_fp8.irpa"
        self.tensor_parallelism_size = 1
        self.dir_path_8b = self.dir_path / "llama-8b"
        self.temp_dir_8b = Path(self.dir_path_8b)
        self.temp_dir_8b.mkdir(parents=True, exist_ok=True)
        self.llama8b_f16_artifacts = ExportArtifacts(
            irpa_path=str(self.irpa_path),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_backends="rocm",
            attention_kernel="decomposed",
            tensor_parallelism_size=self.tensor_parallelism_size,
        )
        self.iree_compile_args = [
            "--iree-hal-target-backends=rocm",
            f"--iree-hip-target={self.iree_hip_target}",
        ]
        self.prefill_args_f16 = self.artifacts_dir / "prefill_args"
        self.decode_args_f16 = self.artifacts_dir / "decode_args"
        self.prefill_args_fp8 = self.artifacts_dir / "prefill_args_fp8"
        self.decode_args_fp8 = self.artifacts_dir / "decode_args_fp8"
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
        output_mlir = self.llama8b_f16_artifacts.create_file(
            suffix=".mlir", prefix=output_file_name
        )
        output_json = self.llama8b_f16_artifacts.create_file(
            suffix=".json", prefix=output_file_name
        )
        output_vmfb = self.llama8b_f16_artifacts.create_file(
            suffix=".vmfb", prefix=output_file_name
        )
        output_shard_file_name = str(
            self.artifacts_dir
            / f"llama3.1_8b_fp16_tp{self.tensor_parallelism_size}_parameters.irpa"
        )
        # shard_irpa file
        shard_return_code = self.llama8b_f16_artifacts.shard_irpa_file(
            output_file=output_shard_file_name
        )
        if shard_return_code == 0:
            self.irpa_path = output_shard_file_name
        export_return_code = self.llama8b_f16_artifacts.export_to_mlir(
            mlir_path=output_mlir,
            json_path=output_json,
        )
        self.llama8b_f16_artifacts.compile_to_vmfb(
            mlir_path=str(output_mlir),
            vmfb_path=output_vmfb,
            hal_dump_path=output_file_name,
        )
        # benchmark prefill
        self.llama8b_f16_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_prefill_args,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama8b_f16_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_decode_args,
            cwd=self.repo_root,
        )

    @longrun
    @is_mi300x
    @pytest.mark.xfail(reason="Export (torch_sdpa)", strict=True, raises=NotImplementedError)
    def testBenchmark8B_f16_Non_Decomposed(self):
        output_file_name = self.dir_path_8b / "f16_torch"
        output_mlir = self.llama8b_f16_artifacts.create_file(
            suffix=".mlir", prefix=output_file_name
        )
        output_json = self.llama8b_f16_artifacts.create_file(
            suffix=".json", prefix=output_file_name
        )
        output_vmfb = self.llama8b_f16_artifacts.create_file(
            suffix=".vmfb", prefix=output_file_name
        )
        self.llama8b_f16_artifacts.attention_kernel = "torch"
        output_shard_file_name = str(
            self.artifacts_dir
            / f"llama3.1_8b_fp16_tp{self.tensor_parallelism_size}_parameters_torch_sdpa.irpa"
        )
        # shard_irpa file
        shard_return_code = self.llama8b_f16_artifacts.shard_irpa_file(
            output_file=output_shard_file_name
        )
        if shard_return_code == 0:
            self.irpa_path = output_shard_file_name
        export_return_code = self.llama8b_f16_artifacts.export_to_mlir(
            mlir_path=output_mlir,
            json_path=output_json,
        )
        self.llama8b_f16_artifacts.compile_to_vmfb(
            mlir_path=str(output_mlir),
            vmfb_path=output_vmfb,
            hal_dump_path=output_file_name,
        )
        # benchmark prefill
        self.llama8b_f16_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_prefill_args,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama8b_f16_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.hip_device_id,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_decode_args,
            cwd=self.repo_root,
        )

    @longrun
    @is_mi300x
    @pytest.mark.xfail(reason="Export (File Not Found)", strict=True, raises=ExportMlirException)
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
    @pytest.mark.xfail(reason="Export (torch_sdpa)", strict=True, raises=NotImplementedError)
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
        artifacts_dir = Path("/data/llama-3.1/70b")
        self.irpa_path = artifacts_dir / "llama70b_f16.irpa"
        self.irpa_path_fp8 = artifacts_dir / "llama70b_fp8.irpa"
        self.tensor_parallelism_size = 1
        self.dir_path_70b = self.dir_path / "llama-70b"
        self.temp_dir_70b = Path(self.dir_path_70b)
        self.temp_dir_70b.mkdir(parents=True, exist_ok=True)
        self.iree_compile_args = [
            "--iree-hal-target-backends=rocm",
            f"--iree-hip-target={self.iree_hip_target}",
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
    @pytest.mark.xfail(reason="Export (File Not Found)", strict=True, raises=ExportMlirException)
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
    @pytest.mark.xfail(reason="Export (torch_sdpa)", strict=True, raises=NotImplementedError)
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
    @pytest.mark.xfail(reason="Export (File Not Found)", strict=True, raises=ExportMlirException)
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
    @pytest.mark.xfail(reason="Export (torch_sdpa)", strict=True, raises=NotImplementedError)
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
        artifacts_dir = Path("/data/llama-3.1/405b")
        self.irpa_path = artifacts_dir / "llama405b_f16.irpa"
        self.irpa_path_fp8 = artifacts_dir / "llama405b_fp8.irpa"
        self.tensor_parallelism_size = 8
        self.dir_path_405b = self.dir_path / "llama-405b"
        self.temp_dir_405b = Path(self.dir_path_405b)
        self.temp_dir_405b.mkdir(parents=True, exist_ok=True)
        self.iree_compile_args = [
            "--iree-hal-target-backends=rocm",
            f"--iree-hip-target={self.iree_hip_target}",
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
    @pytest.mark.xfail(reason="Export", strict=True, raises=ExportMlirException)
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
    @pytest.mark.xfail(reason="Export (torch_sdpa)", strict=True, raises=NotImplementedError)
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
    @pytest.mark.xfail(reason="Export (File Not Found)", strict=True, raises=ExportMlirException)
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
    @pytest.mark.xfail(reason="Export (torch_sdpa)", strict=True, raises=NotImplementedError)
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
