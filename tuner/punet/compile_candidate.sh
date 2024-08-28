#! /usr/bin/env bash

set -eou pipefail

readonly MODE="$1"  # Unused for punet.
readonly INPUT="$2"
readonly DIR="$(dirname "$INPUT")"
readonly BASENAME="$(basename "$INPUT" .mlir)"
readonly OUT="${DIR}/compiled/${BASENAME}.vmfb"

timeout 4s ./punet.sh "$INPUT" -o "$OUT" --compile-from=executable-sources 2>/dev/null || (mv "$INPUT" "$DIR/failed" && exit 1)
tools/iree-dump-module "$OUT" | grep -q 'rocm-hsaco-fb' || (mv "$INPUT" "$DIR/failed" && rm -f "$OUT" && exit 1)
if [ -f "${DIR}/${BASENAME}_config.mlir" ]; then
    cat "${DIR}/../config_prolog.mlir" "${DIR}/${BASENAME}_config.mlir" "${DIR}/../config_epilog.mlir" > "${DIR}/specs/${BASENAME}_spec.mlir"
fi
echo "Compiling ${INPUT}: success"
