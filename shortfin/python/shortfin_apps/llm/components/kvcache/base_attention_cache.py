# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Base class for kv caches.
"""

from typing import List
from attention_paging import PageInfo
import math


class BasePagedAttentionCache:
    """
    Manages lifecycle of pages (using PageInfo as handles).


    Page States:
        Caching - Page can be read by multiple threads
            - Also maintains a reference count
        Writing - Page is being modified by a single owner thread

    Transitions:
        Caching -> Writing: When acquiring an unreferenced LRU leaf page for writing
        Writing -> Caching: When writing is complete and page is released

    Thread Safety:
        - Multiple readers allowed in ReadableCaching state
        - Single writer exclusive access in Writing state
        - Reference counting prevents eviction of in-use pages
    """

    def __init__(self, page_pool, tokens_per_page):
        self.page_pool = page_pool
        self.tokens_per_page = tokens_per_page

    def acquire_pages_for_tokens(
        self, tokens: List[int], extra_token_slots: int = 1
    ) -> tuple[list[PageInfo], int]:
        """
        Given a list of tokens, return a list of pages and a start position to continue generation from.

        Parameters:
        - tokens: all the known tokens for this generation request
        - extra_token_slots: number of kvcache slots needed in addition to the ones needed to hold the given tokens.

        In the base implementation, this will just allocate all new pages, but in shared-kv implementations, we will fetch cached pages if applicable.

        The pages are returned in order.

        No token at idx < n_cached_token should be written to. TODO: consider enforcing this.
        """
        pages_needed = math.ceil(len(tokens + extra_token_slots) / self.tokens_per_page)
        pages = self.page_pool.acquire_free_pages(pages_needed)

        n_cached_tokens = 0

        return pages, n_cached_tokens

    def publish_pages(self, tokens, pages) -> None:
        """
        Given a list of tokens and pages containing KV corresponding to these tokens, make these pages available to other requests.

        Associates the tokens with the pages, and mark them as done writing.

        It is assumed that hereafter, the calling request will not modify these pages, at least not the positions [0:len(tokens)].
        """

        pass  # the base implementation doesn't cache unfinished requests.

    def release_pages(self, tokens, pages):
        """
        Decrement reference count for these pages. When reference count is zero, they will be elegible for eviction.
        """
        # in the base implementation, the pages can be owned by 1 request max, so they can be instantly release
        self.page_pool.release_pages(pages)
