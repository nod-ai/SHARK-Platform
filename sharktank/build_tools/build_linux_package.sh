#!/bin/bash

# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# build_linux_package.sh
#
# Builds sharktank Python package for Linux
#
# Usage:
#   ./build_tools/build_linux_package.sh

set -xeu -o errtrace

THIS_DIR="$(cd $(dirname $0) && pwd)"
REPO_ROOT="$(cd "$THIS_DIR"/../../ && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${THIS_DIR}/wheelhouse}"

python -m pip wheel --disable-pip-version-check -v -w "${OUTPUT_DIR}" "${REPO_ROOT}/sharktank"

wheel_output="$(echo "${OUTPUT_DIR}/sharktank-"*".whl")"
ls "${wheel_output}"

echo "Deleting other wheels"
pushd ${OUTPUT_DIR}
ls | grep -xv "sharktank-"*".whl" | xargs rm
popd
