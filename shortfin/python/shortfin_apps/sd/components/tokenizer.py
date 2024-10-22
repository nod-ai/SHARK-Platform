# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

from transformers import CLIPTokenizer, BatchEncoding

import numpy as np

import shortfin as sf
import shortfin.array as sfnp


class Tokenizer:
    def __init__(
        self,
        raw_tk: CLIPTokenizer,
        max_length: int = 64,
        pad_id: int = 0,
        attn_mask=False,
    ):
        self.pad_id = pad_id
        self._raw = raw_tk
        self.max_length = 64
        self.return_attention_mask = attn_mask

    @staticmethod
    def from_pretrained(name: str, subfolder: str) -> "Tokenizer":
        raw_tk = CLIPTokenizer.from_pretrained(name, subfolder=subfolder)
        return Tokenizer(raw_tk)

    def encode(self, texts: list[str]):
        """Encodes a batch of texts, applying no padding."""
        return self._raw(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="np",
            return_attention_mask=False,
        )

    def encoding_length(self, enc: BatchEncoding) -> int:
        """Gets the length of an encoding."""
        return len(enc.ids)

    def post_process_encodings(self, encs: list[BatchEncoding], batch_seq_len: int):
        """Truncates and pads to a requested size."""
        for enc in encs:
            enc.truncate(batch_seq_len)
            enc.pad(batch_seq_len)

    def encodings_to_array(
        self,
        device: sf.ScopedDevice,
        encs: dict[str, BatchEncoding],
        batch_seq_len: int,
        *,
        dtype: sfnp.DType = sfnp.int32,
    ):
        """Creates a device_array with the contents of a batch of encodings.

        It is expected that the user has called post_process_encodings with
        the same batch_seq_len in order to properly truncate/pad.
        """
        ary = sfnp.device_array.for_host(device, [len(encs), batch_seq_len], dtype)
        for i, enc in enumerate(encs):
            ary.view(i).items = np.expand_dims(enc.input_ids, axis=0)
        return ary

    def attention_masks_to_array(
        self,
        device: sf.ScopedDevice,
        encs: list[BatchEncoding],
        batch_seq_len: int,
        *,
        dtype: sfnp.DType = sfnp.int32,
    ):
        ary = sfnp.device_array.for_host(device, [len(encs), batch_seq_len], dtype)
        for i, enc in enumerate(encs):
            ary.view(i).items = enc.attention_mask
        return ary
