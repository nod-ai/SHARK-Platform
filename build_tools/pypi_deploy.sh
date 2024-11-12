#!/bin/bash

# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script promotes Python packages from nightly releases to PyPI.
#
# Prerequisites:
#   * You will need to have PyPI credentials set up. See
#     https://packaging.python.org/en/latest/tutorials/packaging-projects/#uploading-the-distribution-archives
#   * Install requirements, e.g. in a Python virtual environment (venv):
#     `pip install -r requirements-pypi-deploy.txt`
#   * Choose a release candidate to promote from
#     https://github.com/nod-ai/SHARK-Platform/releases/tag/dev-wheels
#
# Usage:
#   ./pypi_deploy.sh 2.9.0rc20241108

set -euo pipefail

RELEASE="$1"

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";
TMPDIR="$(mktemp --directory --tmpdir shark_platform_pypi_wheels.XXXXX)"
ASSETS_PAGE="https://github.com/nod-ai/SHARK-Platform/releases/expanded_assets/dev-wheels"

# TODO: rewrite in Python?

function download_wheels() {
  echo ""
  echo "Downloading wheels for '${RELEASE}'..."

  # sharktank
  python -m pip download sharktank==${RELEASE} \
    --no-deps --python-version 3.11 -f ${ASSETS_PAGE}

  # shortfin
  python -m pip download shortfin==${RELEASE} \
    --no-deps --python-version 3.11 -f ${ASSETS_PAGE}
  python -m pip download shortfin==${RELEASE} \
    --no-deps --python-version 3.12 -f ${ASSETS_PAGE}
  python -m pip download shortfin==${RELEASE} \
    --no-deps --python-version 3.13 -f ${ASSETS_PAGE}
  python -m pip download shortfin==${RELEASE} \
    --no-deps --python-version 3.13 -f ${ASSETS_PAGE}
  # TODO: 3.13t somehow

  echo ""
  echo "Downloaded wheels:"
  ls
}

function edit_release_versions() {
  echo ""
  echo "Editing release versions..."
  for file in *
  do
    ${SCRIPT_DIR}/promote_whl_from_rc_to_final.py ${file} --delete-old-wheel
  done

  echo "Edited wheels:"
  ls
}

function upload_wheels() {
  echo ""
  echo "Uploading wheels..."
  twine upload --verbose *
}

function main() {
  echo "Changing into ${TMPDIR}"
  cd "${TMPDIR}"
  # TODO: check_requirements (using pip)

  download_wheels
  # edit_release_versions
  # upload_wheels
}

main
