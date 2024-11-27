import pytest
import threading
import queue
import random
import time
from unittest.mock import Mock
from dataclasses import dataclass
from typing import List, Optional

from shortfin_apps.llm.components.kvcache.base_attention_cache import (
    BasePagedAttentionCache,
    BasePageAttentionCacheAllocation,
    CacheAllocationFailure,
)
from shortfin_apps.llm.components.kvcache.page_pool import PagePool, PageInfo


class MockPagePool(PagePool):
    def __init__(self, total_pages: int = 100):
        self._queue = queue.Queue()

        for i in range(total_pages):
            page = PageInfo(index=i, pool=self, token_offset=0, token_count=0)
            self._queue.put(page)

    def acquire_free_pages(self, count: int) -> List[PageInfo]:
        try:
            return [self._queue.get_nowait() for _ in range(count)]
        except queue.Empty:
            return None

    def release_pages(self, pages):
        for page in pages:
            self._queue.put(page)


@pytest.fixture
def page_pool():
    return MockPagePool(total_pages=10)


@pytest.fixture
def cache(page_pool):
    return BasePagedAttentionCache(page_pool=page_pool, tokens_per_page=16)


@pytest.fixture
def page_pool():
    return MockPagePool(total_pages=10)


@pytest.fixture
def cache(page_pool):
    """Create cache with 16 tokens per page"""
    return BasePagedAttentionCache(page_pool=page_pool, tokens_per_page=16)


def test_allocation_sizes(cache):
    test_cases = [
        ([], 0),  # Empty token list
        (list(range(8)), 1),  # Partial page
        (list(range(16)), 1),  # Exact page
        (list(range(17)), 2),  # Just over one page
        (list(range(32)), 2),  # Multiple exact pages
        (list(range(33)), 3),  # Multiple pages with remainder
    ]

    for tokens, expected_pages in test_cases:
        allocation = cache.acquire_pages_for_tokens(tokens)
        pages = allocation.pages
        assert len(pages) == expected_pages
        allocation.release_pages()


def test_concurrent_access(cache):
    def worker(results: List):
        allocation = cache.acquire_pages_for_tokens(list(range(16)))
        results.append(len(allocation.pages))
        allocation.release_pages()

    results = []
    threads = [threading.Thread(target=worker, args=(results,)) for _ in range(5)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r == 1 for r in results)
