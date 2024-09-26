#!/bin/bash
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# build_linux_package.sh
# One stop build of shortfin Python package for Linux. The Linux build is
# complicated because it has to be done via a manylinux docker container.
#
# Usage:
# Build everything (all python versions):
#   sudo ./build_tools/build_linux_package.sh
#
# Build specific Python versions to custom directory:
#   OVERRIDE_PYTHON_VERSIONS="cp312-cp312 cp313-313" \
#   OUTPUT_DIR="/tmp/wheelhouse" \
#   sudo ./build_tools/build_linux_package.sh
#
# Valid Python versions match a subdirectory under /opt/python in the docker
# image. Typically:
#   cp312-cp312 cp313-cp313
#
# Note that this script is meant to be run on CI and it will pollute both the
# output directory and in-tree build/ directories with docker created, root
# owned builds. Sorry - there is no good way around it.
#
# It can be run on a workstation but recommend using a git worktree dedicated
# to packaging to avoid stomping on development artifacts.
set -xeu -o errtrace

THIS_DIR="$(cd $(dirname $0) && pwd)"
REPO_ROOT="$(cd "$THIS_DIR"/../../ && pwd)"
SCRIPT_NAME="$(basename $0)"
ARCH="$(uname -m)"

# TODO: update to manylinux_2_28, 2014 is being used by torch-mlir
# MANYLINUX_DOCKER_IMAGE="${MANYLINUX_DOCKER_IMAGE:-"ghcr.io/nod-ai/manylinux_x86_64:main@sha256:facedb71df670016e74e646d71e869e6fff70d4cdbaa6634d4d0a10d6e174399" }')}"
MANYLINUX_DOCKER_IMAGE="${MANYLINUX_DOCKER_IMAGE:-quay.io/pypa/manylinux2014_${ARCH}@sha256:2ace4b4f06a726d270821cb4993caeed3aacdaa54801e9f13be8fbe96077f4dc}"
PYTHON_VERSIONS="${OVERRIDE_PYTHON_VERSIONS:-cp312-cp312}"
OUTPUT_DIR="${OUTPUT_DIR:-${THIS_DIR}/wheelhouse}"

function run_on_host() {
  echo "Running on host"
  echo "Launching docker image ${MANYLINUX_DOCKER_IMAGE}"

  # Canonicalize paths.
  mkdir -p "${OUTPUT_DIR}"
  OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
  echo "Outputting to ${OUTPUT_DIR}"
  mkdir -p "${OUTPUT_DIR}"
  docker run --rm \
    -v "${REPO_ROOT}:${REPO_ROOT}" \
    -v "${OUTPUT_DIR}:${OUTPUT_DIR}" \
    -e __MANYLINUX_BUILD_WHEELS_IN_DOCKER=1 \
    -e "OVERRIDE_PYTHON_VERSIONS=${PYTHON_VERSIONS}" \
    -e "OUTPUT_DIR=${OUTPUT_DIR}" \
    "${MANYLINUX_DOCKER_IMAGE}" \
    -- ${THIS_DIR}/${SCRIPT_NAME}

  echo "******************** BUILD COMPLETE ********************"
  echo "Generated binaries:"
  ls -l "${OUTPUT_DIR}"
}

function run_in_docker() {
  echo "Running in docker"
  echo "Marking git safe.directory"
  git config --global --add safe.directory '*'

  echo "Using python versions: ${PYTHON_VERSIONS}"
  local orig_path="${PATH}"

  # Build phase.
  echo "******************** BUILDING PACKAGE ********************"
  for python_version in ${PYTHON_VERSIONS}; do
    python_dir="/opt/python/${python_version}"
    if ! [ -x "${python_dir}/bin/python" ]; then
      echo "ERROR: Could not find python: ${python_dir} (skipping)"
      continue
    fi
    export PATH="${python_dir}/bin:${orig_path}"
    echo ":::: Python version $(python --version)"
    clean_wheels "shortfin" "${python_version}"
    build_shortfin
    run_audit_wheel "shortfin" "${python_version}"
  done
}

function build_shortfin() {
  export SHORTFIN_ENABLE_TRACING=ON
  python -m pip wheel --disable-pip-version-check -v -w "${OUTPUT_DIR}" "${REPO_ROOT}/libshortfin"
}

function run_audit_wheel() {
  local wheel_basename="$1"
  local python_version="$2"
  generic_wheel="${OUTPUT_DIR}/${wheel_basename}-${python_version}-linux_$(uname -m).whl"
  ls "${generic_wheel}"
  echo ":::: Auditwheel ${generic_wheel}"
  auditwheel repair -w "${OUTPUT_DIR}" "${generic_wheel}"
  rm -v "${generic_wheel}"
}

function clean_wheels() {
  local wheel_basename="$1"
  local python_version="$2"
  echo ":::: Clean wheels ${wheel_basename} ${python_version}"
  rm -f -v "${OUTPUT_DIR}/${wheel_basename}-"*"-${python_version}-"*".whl"
}

# Trampoline to the docker container if running on the host.
if [ -z "${__MANYLINUX_BUILD_WHEELS_IN_DOCKER-}" ]; then
  run_on_host "$@"
else
  run_in_docker "$@"
fi
