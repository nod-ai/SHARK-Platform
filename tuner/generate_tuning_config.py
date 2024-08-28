# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input tuning spec, generates the config files used by the tuner.

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input spec file", type=str)

    args = parser.parse_args()
    input_path: str = args.input

    lines: list[str] = []
    with open(input_path, "r") as f:
        lines = f.readlines()

    config_baseline_lines: list[str] = []
    config_prolog_lines: list[str] = []
    config_epilog_lines: list[str] = []

    found_tuning_spec_begin = False
    found_tuning_spec_end = False
    found_tuning_match_begin = False
    found_tuning_match_end = False

    for i, line in enumerate(lines):
        if "TUNING_SPEC_BEGIN" in line:
            found_tuning_spec_begin = True
            lines = lines[i + 1 :]
            break

        config_baseline_lines.append(line)
        config_prolog_lines.append(line)

    for i, line in enumerate(lines):
        if "TUNING_SPEC_END" in line:
            found_tuning_spec_end = True
            lines = lines[i + 1 :]
            break

    for i, line in enumerate(lines):
        if "TUNING_MATCH_BEGIN" in line:
            found_tuning_match_begin = True
            lines = lines[i + 1 :]
            break

        config_baseline_lines.append(line)
        config_epilog_lines.append(line)

    config_epilog_lines.append("        , @match_op -> @apply_op_config\n")

    for i, line in enumerate(lines):
        if "TUNING_MATCH_END" in line:
            found_tuning_match_end = True
            lines = lines[i + 1 :]
            break

    config_baseline_lines += lines
    config_epilog_lines += lines

    assert found_tuning_spec_begin
    assert found_tuning_spec_end
    assert found_tuning_match_begin
    assert found_tuning_match_end

    with open("config_baseline.mlir", "w") as f:
        f.writelines(config_baseline_lines)

    with open("config_prolog.mlir", "w") as f:
        f.writelines(config_prolog_lines)

    with open("config_epilog.mlir", "w") as f:
        f.writelines(config_epilog_lines)

    return 0


if __name__ == "__main__":
    exit(main())
