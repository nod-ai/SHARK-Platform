#! /usr/bin/env bash

set -xeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
readonly INPUT="$(realpath "$1")"
shift 1

"${SCRIPT_DIR}/../int8-model/compile-punet-base.sh" "${SCRIPT_DIR}/tools/iree-compile" gfx942 \
  "${SCRIPT_DIR}/config.mlir" \
  "$INPUT" \
  "$@"

# --iree-hal-dump-executable-files-to=dump-unet \
