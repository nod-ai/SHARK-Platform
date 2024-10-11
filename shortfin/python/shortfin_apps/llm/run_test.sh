#!/bin/bash
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -xeuo pipefail

mkdir -p /tmp/sharktank/llama

huggingface-cli download --local-dir /tmp/sharktank/llama SlyEcho/open_llama_3b_v2_gguf open-llama-3b-v2-f16.gguf

HUGGING_FACE_TOKENIZER="openlm-research/open_llama_3b_v2"

python - <<EOF
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("${HUGGING_FACE_TOKENIZER}")
tokenizer.save_pretrained("/tmp/sharktank/llama")
EOF

python -m sharktank.examples.export_paged_llm_v1 \
  --gguf-file="/tmp/sharktank/llama/open-llama-3b-v2-f16.gguf" \
  --output-mlir="/tmp/sharktank/llama/model.mlir" \
  --output-config="/tmp/sharktank/llama/config.json"

iree-compile "/tmp/sharktank/llama/model.mlir" \
  --iree-hal-target-backends=rocm \
  --iree-hip-target=gfx1100 \
  -o /tmp/sharktank/llama/model.vmfb

# Write the JSON configuration to edited_config.json
cat > /tmp/sharktank/llama/edited_config.json << EOF
{
    "module_name": "module",
    "module_abi_version": 1,
    "max_seq_len": 2048,
    "attn_head_count": 32,
    "attn_head_dim": 100,
    "prefill_batch_sizes": [
        4
    ],
    "decode_batch_sizes": [
        4
    ],
    "transformer_block_count": 26,
    "paged_kv_cache": {
        "block_seq_stride": 16,
        "device_block_count": 256
    }
}
EOF

# Start the server in the background and save its PID
python -m shortfin_apps.llm.server \
  --tokenizer=/tmp/sharktank/llama/tokenizer.json \
  --model_config=/tmp/sharktank/llama/edited_config.json \
  --vmfb=/tmp/sharktank/llama/model.vmfb \
  --parameters=/tmp/sharktank/llama/open-llama-3b-v2-f16.gguf \
  --device=hip &

SERVER_PID=$!

# Wait a bit for the server to start up
sleep 5

# Run the client
python client.py

# Kill the server
kill $SERVER_PID

# Wait for the server to shut down
wait $SERVER_PID