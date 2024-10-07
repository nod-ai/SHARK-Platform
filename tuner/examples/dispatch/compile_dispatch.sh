#! /usr/bin/env bash

set -eou pipefail

readonly INPUT="$1"
readonly DIR="$(dirname "$INPUT")"
readonly BASENAME="$(basename "$INPUT" .mlir)"
readonly OUT="${DIR}/compiled/${BASENAME}.vmfb"

iree-compile "$INPUT" -o "$OUT" \
  --compile-from=executable-sources 2>/dev/null || (mv "$INPUT" "$DIR/failed" && exit 1)

iree-dump-module "$OUT" | grep -q 'rocm-hsaco-fb' || (mv "$INPUT" "$DIR/failed" && rm -f "$OUT" && exit 1)
if [ -f "${DIR}/${BASENAME}_config.mlir" ]; then
    cat "${DIR}/../config_prolog.mlir" "${DIR}/${BASENAME}_config.mlir" "${DIR}/../config_epilog.mlir" > "${DIR}/specs/${BASENAME}_spec.mlir"
fi

echo "Compiling ${INPUT}: success"
