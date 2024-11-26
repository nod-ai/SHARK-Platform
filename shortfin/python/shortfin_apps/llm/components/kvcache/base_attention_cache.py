# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Base class for kv caches.
"""

from typing import List, Iterable, Protocol
from .page_pool import PageInfo
import math
from abc import ABC, abstractmethod
from .page_pool import PagePool

# logging
import logging

logger = logging.getLogger(__name__)

# exception for when cache allocation failed
class CacheAllocationFailure(Exception):
    pass


class PageAllocation(ABC):
    """
    Abstract base class for page allocations in the cache.
    Subclasses only need to implement the core allocation methods.
    """

    @abstractmethod
    def get_page_list(self) -> List[PageInfo]:
        """Returns the list of pages that were allocated."""
        pass

    @abstractmethod
    def publish_pages(self, up_to_page_index) -> None:
        """
        Makes self.get_page_list()[0:up_to_page_index] available to other requests after writing is complete.
        Associates tokens with pages and marks them as ready for reading.
        """
        pass

    @abstractmethod
    def release_pages(self) -> None:
        """
        Releases the allocation's reference to pages.
        Pages become eligible for eviction when their reference count reaches zero.
        """
        pass


class BasePageAttentionCacheAllocation(PageAllocation):
    """
    Represents a page allocation in the cache, implementing the PageAllocation protocol.
    """

    def __init__(self, pages: Iterable[PageInfo], cache: "BasePagedAttentionCache"):
        # this should only be called by the associated attention cache &
        self._pages = tuple(pages)
        self._cache = cache
        self._is_released = False

    def get_page_list(self) -> List[PageInfo]:
        return list(self._pages)  # return a list, as expected by service.py

    def publish_pages(self, up_to_page_index) -> None:
        """
        Release self.get_pages_list()[0:up_to_page_index] for reading by other requests.

        This should be called when writing completes, after each kernel invocation.
        """
        pass  # the base implementation doesn't cache unfinished requests.

    def release_pages(self) -> None:
        """
        Decrement reference count for these pages. When reference count is zero, they will be elegible for eviction.

        This should be called when the request has finished reading from the pages, and they are no longer needed.

        This does not immediately release the pages, but decrements the reference count.

        Pages should become available for eviction when their reference count reaches zero & the pool runs out of free pages.
        """
        # in the base implementation, the pages can be owned by 1 request max, so they can be instantly release
        if self._is_released:
            logger.warning("Releasing already-released allocation")
            return
        self._cache.page_pool.release_pages(self._pages)
        self._is_released = True

    def __repr__(self):
        return f"BasePageAttentionCacheAllocation(pages={self._pages}, cache={self._cache})"


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

    def __init__(self, page_pool: PagePool, tokens_per_page: int):
        self.page_pool = page_pool
        self.tokens_per_page = tokens_per_page

    def acquire_pages_for_tokens(
        self, tokens: List[int], extra_token_slots: int = 1
    ) -> PageAllocation:
        """
        Given a list of tokens, return a list of pages and a start position to continue generation from.

        Parameters:
        - tokens: all the known tokens for this generation request
        - extra_token_slots: number of kvcache slots needed in addition to the ones needed to hold the given tokens.

        In the base implementation, this will just allocate all new pages, but in shared-kv implementations, we will fetch cached pages if applicable.

        The pages are returned in order.

        No token at idx < n_cached_token should be written to. TODO: consider enforcing this.
        """
        token_count = len(tokens)
        pages_needed = math.ceil(token_count / self.tokens_per_page)
        pages = self.page_pool.acquire_free_pages(pages_needed)

        if pages is None:
            raise CacheAllocationFailure()

        n_cached_tokens = 0

        return BasePageAttentionCacheAllocation(pages, cache=self)
