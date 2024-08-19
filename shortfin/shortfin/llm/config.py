# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Configuration objects.

Parameters that are intrinsic to a specific model.

In a typical transformer model, the KV cache is organized similar to (mapped to
our parameter names below):
    k = tensor.empty(transformer_block_count, batch_size, seq,
                    attn_head_count, attn_head_dim)
    v = ...

For context, a popular model has parameters of:
    attn_dtype_size = 2  # (fp16)
    max_seq_len = 2048
    transformer_block_count = 32
    attn_head_count = 32
    attn_head_dim = 128   # (dim / head_count)

If paging, then we primary care about the organization of a single block, where
a block represents a single position in the sequence for a single item in the batch.
Therefore, it will be organized like:
    block = torch.empty(transformer_block_count, 2, attn_head_count, attn_head_dim)

In this scenario, we declare that one block holds the KV cache for all transformer
block layers because it reduces the accounting. As such, for the above example,
a single position in the sequence will be 524,288 bytes, assuming a 2-byte element
type. If we choose to block by block_stride=16 positions, each block will be 8MiB.
Assuming we wanted to dedicate 12GiB to the block cache, this would equate to 1536
blocks for a total number of sequence positions of 24,576.

These are well-known numbers but are derived above to give a sense of scale.

In order to indirect through to the block cache, we have to provide the index map
to specific invocations:

* Prefill: Prefill is only writing to the blocks from [0:prompt_len], so it will
    need write indices of [batch_size, prompt_len // block_stride + 1].
* Decode step: Decode is auto-regressive, and needs to first compute the new kv
    row and then attend over all rows in the cache up to this point in the sequence.

If wanting to avoid dynamic allocation of transients, we can also pool the index
tables based on the maximum batch size and maximum sequence length. Since all
block cache sizes are well within the range of an i16, we will use that for storage.
Therefore, each batch invocation would need a block lookup table of:

    byte_size = max_batch_size * (max_seq_len // block_stride) * sizeof(int16_t)

For a max_batch_size of 16, this is 4KiB of block index table lookups per
invocation. We don't have to statically allocate this, but the system is more
predictable if we just reserve what we need. Again, numbers are given to give a
sense of scale only: real workloads will vary.
"""

from dataclasses import dataclass

from iree.runtime import (  # type: ignore
    HalElementType,
)

import json


@dataclass
class ModelParams:
    """Parameters for a specific compiled model, sufficient to do cache planning and
    invocations."""

    # The element type of the attention caches.
    attn_dtype: HalElementType

    # Maximum length of a sequence including prompt and output.
    max_seq_len: int

    # Number of transformer blocks.
    transformer_block_count: int

    # Number of attention heads per block.
    attn_head_count: int

    # Dimensionality of each attention head
    attn_head_dim: int

    # Position stride per attention block
    block_seq_stride: int

    # Batch sizes that the prefill stage is compiled for. These are expected to be
    # functions exported from the model with suffixes of "_bs{batch_size}". Must
    # be in ascending order.
    prefill_batch_sizes: list[int]

    # Similarly, batch sizes that the decode stage is compiled for.
    decode_batch_sizes: list[int]

    # Name of the IREE module implementing the model.
    module_name: str = "module"

    # ABI of the module.
    module_abi_version: int = 1

    # Size in bytes of the KV cache dtype.
    @property
    def attn_dtype_size(self) -> int:
        assert HalElementType.is_byte_aligned(self.attn_dtype)
        return HalElementType.dense_byte_count(self.attn_dtype)

    @property
    def max_prefill_batch_size(self) -> int:
        return self.prefill_batch_sizes[-1]

    @property
    def max_decode_batch_size(self) -> int:
        return self.decode_batch_sizes[-1]

    @property
    def max_batch_size(self):
        return max(self.max_prefill_batch_size, self.max_decode_batch_size)

    @staticmethod
    def load_json(path):
        f = open(path)
        j = json.load(f)
        return ModelParams(attn_dtype=HalElementType.FLOAT_16, **j)


@dataclass
class CacheParams:
    """Parameters for management of the block cache.

    This is paired with a ModelParams.

    We presently use a static block cache configuration and hand-wave either a tuning
    run or pen/paper analysis to derive the parameters.
    """

    model: ModelParams

    # The size of the static block cache on the device.
    device_block_count: int

    # The stride of each block in sequence positions.
    block_pos_stride: int

    @property
    def attn_unit_size_elements(self) -> int:
        """Size in bytes of each cache line in the attention cache.

        Each cache line can store a unit position stride.
        """
        size = 1
        size *= self.model.transformer_block_count
        size *= 2  # K and V cache line
        size *= self.model.attn_head_count
        size *= self.model.attn_head_dim
        return size

    @property
    def attn_block_size_elements(self) -> int:
        """Size in bytes of each attention block of {block_position_stride} positions."""
        return self.attn_unit_size_elements * self.block_pos_stride


@dataclass
class ServiceParams:
    """Parameters for the serving service."""

    cache: CacheParams
    model: ModelParams


# From: https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
def human_size(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"
