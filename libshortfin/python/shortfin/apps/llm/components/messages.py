# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum

import shortfin as sf
import shortfin.array as sfnp

from .cache import AttnPageCache, AttnPageEntry


class InferencePhase(Enum):
    PREFILL = 1
    DECODE = 2


class InferenceExecRequest(sf.Message):
    """Performs a prefill operation."""

    def __init__(self, phase: InferencePhase, input_token_ids: list[int]):
        super().__init__()
        self.phase = phase
        self.start_position: int = 0
        self.input_token_ids = input_token_ids
        self.done = sf.VoidFuture()

        # Response control.
        # If True, return all sequence position logits. If False, return only
        # the last.
        self.return_all_logits: bool = False

        # Move the result array to the host and sync to ensure data is
        # available.
        self.return_host_array: bool = True

        # Result logits as [1, sl, d] where 1 is the preserved batch dim,
        # sl is either 1 (not return_all_logits) or >=1 (return_all_logits).
        self.result_logits: sfnp.device_array | None = None

        # Cache pages that have been locked for this request.
        self._cache: AttnPageCache | None = None
        self.locked_pages: list[AttnPageEntry] | None = None

    def reset(self, phase: InferencePhase):
        """Resets all per request state in preparation for an subsequent execution."""
        self.phase = phase
        self.done = sf.VoidFuture()
        self.return_all_logits = False
        self.return_host_array = True
        self.result_logits = None

    def cache_page_indices(self, max_len: int) -> list[int]:
        if not self.locked_pages:
            return []
        indices = [p.index for p in self.locked_pages]
        if len(indices) > max_len:
            return indices[0:max_len]
        return indices

    def free_cache_pages(self):
        cache = self._cache
        if cache:
            pages = self.locked_pages
            self._cache = None
            self.locked_pages = None
            cache.release_pages(pages)

    def lock_initial_cache_pages(
        self, cache: AttnPageCache, pages: list[AttnPageEntry]
    ):
        assert not self._cache
        self._cache = cache
        self.locked_pages = pages

    def lock_new_cache_pages(self, cache: AttnPageCache, pages: list[AttnPageEntry]):
        assert self._cache is cache
        self.locked_pages.extend(pages)


class StrobeMessage(sf.Message):
    """Sent to strobe a queue with fake activity (generate a wakeup)."""

    ...
