# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from tuner import libtuner


def main():
    args = libtuner.parse_arguments()

    path_config = libtuner.PathConfig()
    path_config.base_dir.mkdir(parents=True, exist_ok=True)
    path_config.output_unilog.touch()
    candidate_trackers: list[libtuner.CandidateTracker] = []
    stop_after_phase: str = args.stop_after

    print("Setup logging")
    libtuner.setup_logging(args, path_config)
    print(path_config.run_log, end="\n\n")

    if not args.dry_run:
        print("Validating devices")
        libtuner.validate_devices(args.devices)
        print("Validation successful!\n")

    print("Generating candidates...")
    candidates = libtuner.generate_candidate_specs(
        args, path_config, candidate_trackers
    )
    print(f"Stored candidate specs in {path_config.specs_dir}\n")
    if stop_after_phase == libtuner.ExecutionPhases.generate_candidates:
        return

    print("Check the detailed execution logs in:")
    print(path_config.run_log.resolve())

    for candidate in candidate_trackers:
        libtuner.logging.debug(candidate)
