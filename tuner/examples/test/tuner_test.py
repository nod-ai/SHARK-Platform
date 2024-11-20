# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Sample Usage:

python -m examples.test double_mmt.mlir mmt_benchmark.mlir --devices=hip://0,hip://1 --num-candidates=64


Recommended Trial Run:

python -m examples.test double_mmt.mlir mmt_benchmark.mlir --num-candidates=10


Dry Run Test (no gpu required, not currently working):

python -m examples.test double_mmt.mlir mmt_benchmark.mlir --num-candidates=64 --dry-run

"""

from tuner import libtuner
from pathlib import Path, PurePath
import argparse
import os


class TestTuner(libtuner.TuningClient):
    def __init__(self):
        self.compile_flags = [
            "--iree-hip-target=gfx942",
            "--compile-from=executable-sources",
        ]
        self.benchmark_flags = [
            "--benchmark_repetitions=3",
            "--benchmark_format=json",
        ]

    def get_compile_timeout_s(self) -> int:
        return 10

    def get_iree_compile_flags(self) -> list[str]:
        return self.compile_flags

    def get_benchmark_timeout_s(self) -> int:
        return 15

    def get_iree_benchmark_module_flags(
        self,
    ) -> list[str]:
        return self.benchmark_flags

    # TODO(Max191): Remove the following unused abstract functions once they
    # are removed from the TuningClient definition.
    def get_dispatch_benchmark_timeout_s(self) -> int:
        return 0

    def get_dispatch_compile_timeout_s(self) -> int:
        return 0

    def get_dispatch_compile_command(
        self, candidate_tracker: libtuner.CandidateTracker
    ) -> list[str]:
        return []

    def get_dispatch_benchmark_command(
        self,
        candidate_tracker: libtuner.CandidateTracker,
    ) -> list[str]:
        return []

    def get_model_compile_timeout_s(self) -> int:
        return 0

    def get_model_compile_command(
        self, candidate_tracker: libtuner.CandidateTracker
    ) -> list[str]:
        return []

    def get_model_benchmark_timeout_s(self) -> int:
        return 0

    def get_model_benchmark_command(
        self, candidate_tracker: libtuner.CandidateTracker
    ) -> list[str]:
        return []


def main():
    # Custom arguments for the test file.
    parser = argparse.ArgumentParser(description="Autotune test script")
    test_args = parser.add_argument_group("Example Test Options")
    test_args.add_argument(
        "model_file", type=Path, help="Path to the model file to benchmark (.mlir)"
    )
    # Remaining arguments come from libtuner
    args = libtuner.parse_arguments(parser)

    path_config = libtuner.PathConfig()
    path_config.base_dir.mkdir(parents=True, exist_ok=True)
    path_config.output_unilog.touch()
    candidate_trackers: list[libtuner.CandidateTracker] = []
    test_tuner = TestTuner()
    stop_after_phase: str = args.stop_after

    print("Setup logging")
    libtuner.setup_logging(args, path_config)
    print(path_config.run_log, end="\n\n")

    if not args.dry_run:
        print("Validating devices")
        libtuner.validate_devices(args.devices)
        print("Validation successful!\n")

    print("Generating candidates...")
    candidates = libtuner.generate_candidate_specs(args, path_config, candidate_trackers)
    print(f"Stored candidates in {path_config.candidates_dir}\n")
    if stop_after_phase == libtuner.ExecutionPhases.generate_candidates:
        return

    print("Compiling candidates...")
    compiled_candidates = libtuner.compile(
        args, path_config, candidates, candidate_trackers, test_tuner
    )
    print(f"Compiled files are stored in {path_config.compiled_dir}\n")
    if stop_after_phase == libtuner.ExecutionPhases.compile_dispatches:
        return

    print("Benchmarking compiled candidates...")
    top_candidates = libtuner.benchmark(
        args, path_config, compiled_candidates, candidate_trackers, test_tuner
    )
    print(f"\nStored kernel results in {path_config.output_unilog.resolve()}\n")
    if stop_after_phase == libtuner.ExecutionPhases.benchmark_dispatches:
        return

    print("Compiling model with top candidates...")
    test_tuner.compile_flags = [
        "--iree-hip-target=gfx942",
    ]
    compiled_model_candidates = libtuner.compile(
        args, path_config, candidates, candidate_trackers, test_tuner, args.model_file
    )

    print("Benchmarking compiled model candidates...")
    test_tuner.benchmark_flags.append("--input=2048x2048xf16")
    test_tuner.benchmark_flags.append("--input=2048x2048xf16")
    test_tuner.benchmark_flags.append("--function=main")
    top_candidates = libtuner.benchmark(
        args, path_config, compiled_candidates, candidate_trackers, test_tuner
    )
    print(f"\nStored model results in {path_config.output_unilog.resolve()}\n")
    if stop_after_phase == libtuner.ExecutionPhases.benchmark_models:
        return
    
    libtuner.save_pickle(path_config.candidate_trackers_pkl, candidate_trackers)
    print(f"Candidate trackers are saved in {path_config.candidate_trackers_pkl}\n")

    print("Check the detailed execution logs in:")
    print(path_config.run_log.resolve())

    for candidate in candidate_trackers:
        libtuner.logging.debug(candidate)
