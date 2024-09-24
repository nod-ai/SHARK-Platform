# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Sequence

import logging
import math
import threading

import shortfin as sf

from .config_struct import ModelParams, human_size

logger = logging.getLogger(__name__)


class AttnPageEntry:
    __slots__ = [
        "cache",
        "index",
        "in_use",
    ]

    def __init__(self, cache: "AttnPageCache", index: int):
        self.cache = cache
        self.index = index
        self.in_use = False

    def __repr__(self):
        return f"Block({self.index}, {'FREE' if not self.in_use else 'BUSY'})"


class AttnPageCache:
    """Page table based attention cache.

    While internal to a model, the cache is organized with additional structure
    per page, outside of the model, it is just a list of pages of a certain
    element type and number of elements (all inner dims are flattened).

    One page table is allocated per device in a fiber. Currently, this is a
    dense allocation with committed memory but in the future, we may just
    allocate the address space and lazily populate it with committed memory.

    The cache is unique because usage of it can span fibers and concurrency
    is implicitly managed at the block level (i.e. freshly acquired blocks
    are assumed to be uninitialized and available immediately for use).

    It is initialized with a discrete list of fiberd devices from a fiber but
    cache usage can be done from any fiber which includes those devices.
    """

    def __init__(
        self, *, devices: Sequence[sf.ScopedDevice], model_params: ModelParams
    ):
        self._lock = threading.Lock()
        self.devices = list(devices)
        self.model_params = model_params
        self.page_tables: list[sf.array.device_array] = []
        cache_params = model_params.paged_kv_cache
        alloc_page_count = cache_params.device_block_count

        # Setup accounting structs.
        self.attn_page_entries = [
            AttnPageEntry(self, i) for i in range(alloc_page_count)
        ]
        self.attn_page_free = list(self.attn_page_entries)

        # Initialize a page table on each device.
        assert cache_params is not None, "Model does not have a paged kv cache"
        page_table_shape = [
            alloc_page_count,
            model_params.paged_kv_block_size_elements,
        ]
        for device in devices:
            logging.info(
                "Allocating page table (shape=%r, dtype=%r, size=%s) on %r",
                page_table_shape,
                model_params.attn_dtype,
                human_size(
                    math.prod(page_table_shape)
                    * model_params.attn_dtype.dense_byte_count
                ),
                device,
            )
            page_table = sf.array.device_array.for_device(
                device, page_table_shape, model_params.attn_dtype
            )
            self.page_tables.append(page_table)

    def acquire_free_pages(self, count: int) -> list[AttnPageEntry] | None:
        with self._lock:
            available = len(self.attn_page_free)
            if count > available:
                return None
            return [self.attn_page_free.pop() for _ in range(count)]

    def release_pages(self, pages: list[AttnPageEntry]):
        with self._lock:
            self.attn_page_free.extend(pages)

    def __repr__(self):
        # No need to lock for repr (list is internally synchronized).
        free_pages = len(self.attn_page_free)
        total_pages = len(self.attn_page_entries)
        return (
            f"AttnPageCache({total_pages - free_pages}/{total_pages} pages in use: "
            f"{100.0 * free_pages / total_pages}% free)"
        )
