#!/bin/bash
# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -xeuo pipefail

# Assume that the environment is already set up:
#   * Python venv set up with requirements, sharktank, and shortfin
#   * iree-compile and iree-run-module on $PATH
#   * authenticated with `huggingface-cli login`

# Input variables.
#   Default model: https://huggingface.co/SlyEcho/open_llama_3b_v2_gguf
#   Default tokenizer: https://huggingface.co/openlm-research/open_llama_3b_v2
TEMP_DIR="${TEMP_DIR:-/tmp/sharktank/llama}"
HUGGING_FACE_MODEL_NAME="${HUGGING_FACE_MODEL_NAME:-SlyEcho/open_llama_3b_v2_gguf}"
HUGGING_FACE_MODEL_FILE="${HUGGING_FACE_MODEL_FILE:-open-llama-3b-v2-f16.gguf}"
HUGGING_FACE_TOKENIZER="${HUGGING_FACE_TOKENIZER:-openlm-research/open_llama_3b_v2}"

# Derived variables.
LOCAL_GGUF_FILE="${TEMP_DIR}/${HUGGING_FACE_MODEL_FILE}"
LOCAL_MLIR_FILE="${TEMP_DIR}/model.mlir"
LOCAL_CONFIG_FILE="${TEMP_DIR}/config.json"
LOCAL_VMFB_FILE="${TEMP_DIR}/model.vmfb"

mkdir -p ${TEMP_DIR}

huggingface-cli download --local-dir ${TEMP_DIR} ${HUGGING_FACE_MODEL_NAME} ${HUGGING_FACE_MODEL_FILE}

python -m sharktank.examples.export_paged_llm_v1 \
  --gguf-file="${LOCAL_GGUF_FILE}" \
  --output-mlir="${LOCAL_MLIR_FILE}" \
  --output-config="${LOCAL_CONFIG_FILE}"

iree-compile "${LOCAL_MLIR_FILE}" \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu-features=host \
  -o ${LOCAL_VMFB_FILE}

python -m shortfin.llm.impl.service_v1_cli \
  --tokenizer="${HUGGING_FACE_TOKENIZER}" \
  --config="${LOCAL_CONFIG_FILE}" \
  --vmfb="${LOCAL_VMFB_FILE}" \
  --gguf="${LOCAL_GGUF_FILE}"
