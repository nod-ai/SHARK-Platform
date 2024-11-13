from __future__ import annotations
from typing import List, Tuple, Optional, Sequence
import threading
import logging
import shortfin as sf
from dataclasses import dataclass

from ..config_struct import human_size
import math

import time

logger = logging.getLogger(__name__)


@dataclass
class PageInfo:
    """
    Page index with some metadata about its contents.
    """

    page_index: int
    in_use: bool
    pool: PagePool
    token_offset: int  # Offset within the page
    token_count: int  # Number of tokens stored in this page
    ref_count: int = 0  # Number of references to this page in the radix tree


@dataclass
class PagePoolConfig:
    """
    Hyperparameters for the page pool.
    """

    device_block_count: int
    dtype: sf.dtype
    alloc_page_count: int

    paged_kv_block_size_elements: int  # size of a single page as # of elements
    # (e.g. one configuration for llama3.1 8b hax 32x2x16x8x128=1048576 elements where:
    # 32: number of transformer blocks
    # 2: one for k + one for v
    # 16: tokens per page
    # 8: head count (32 heads, but every 4 heads share the same kv buffer)
    # 128: hidden dimension


class PagePool:
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

    def __init__(self, *, devices: Sequence[sf.ScopedDevice], config: PagePoolConfig):
        self._lock = threading.Lock()
        self.devices = list(devices)
        self.config = config
        self.page_tables: list[sf.array.device_array] = []

        # Setup accounting structs.
        self.attn_page_entries = [
            PageInfo(
                page_index=i,
                in_use=False,
                pool=self,
                token_offset=0,
                token_count=0,
                ref_count=0,
            )
            for i in range(self.config.alloc_page_count)
        ]

        self.attn_page_free = list(self.attn_page_entries)

        # Initialize a page table on each device.
        page_table_shape = [
            self.config.alloc_page_count,
            self.config.paged_kv_block_size_elements,
        ]
        for device in devices:
            logging.info(
                "Allocating page table (shape=%r, dtype=%r, size=%s) on %r",
                page_table_shape,
                self.config.dtype,
                human_size(
                    math.prod(page_table_shape) * self.config.dtype.dense_size_bytes
                ),
                device,
            )
            page_table = sf.array.device_array.for_device(
                device, page_table_shape, self.config.dtype
            )
            self.page_tables.append(page_table)

    def acquire_free_pages(self, count: int) -> list[PageInfo] | None:
        with self._lock:
            available = len(self.attn_page_free)
            if count > available:
                return None
            return [self.attn_page_free.pop() for _ in range(count)]

    def release_pages(self, pages: list[PageInfo]):
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


############################## begin radix attention


@dataclass
class RadixNode:
    """Node in radix tree tracking pages"""

    children: dict[int, RadixNode]
    parent: Optional[RadixNode]
    key: List[int]
    pages: List[PageInfo]
    last_access_timestamp: int = 0
    ref_count: int = 0


class RadixTree:
    """
    Radix Tree for mapping token sequences to pages in the attention cache.

    Requests pages from a PagePool to store kvs for tokens in the sequence.
    """

    def __init__(
        self, *, page_pool: PagePool, tokens_per_page: int, disable: bool = False
    ):
        self._lock = threading.Lock()
        self.page_pool = page_pool
        self.disable = disable
        self.tokens_per_page = tokens_per_page
        self.reset()

    def reset(self) -> None:
        """Reset the cache state"""
        with self._lock:
            # free
            self.root = RadixNode(
                children={}, parent=None, key=[], pages=[], ref_count=1
            )

    def _get_match_len(self, key1: List[int], key2: List[int]) -> int:
        """Return length of matching prefix between two keys"""
        for i, (k1, k2) in enumerate(zip(key1, key2)):
            if k1 != k2:
                return i
        return min(len(key1), len(key2))

    def match_prefix(self, token_ids: List[int]) -> Tuple[List[PageInfo], RadixNode]:
        """Find longest matching prefix and return its pages"""
        if self.disable:
            return [], self.root

        with self._lock:
            matched_pages = []
            last_node = self.root
            curr_node = self.root
            remaining_tokens = token_ids

            while remaining_tokens:
                first_token = remaining_tokens[0]
                if first_token not in curr_node.children:
                    break

                child = curr_node.children[first_token]
                match_len = self._get_match_len(child.key, remaining_tokens)

                if match_len < len(child.key):
                    # Partial match - need to split
                    new_node = self._split_node(child, match_len)
                    matched_pages.extend(new_node.pages)
                    last_node = new_node
                    break
                else:
                    # Full match of this node
                    matched_pages.extend(child.pages)
                    last_node = child
                    remaining_tokens = remaining_tokens[match_len:]
                    curr_node = child

            # Update access time and ref counts
            self._update_access_path(last_node)
            for page in matched_pages:
                page.ref_count += 1

            return matched_pages, last_node

    def _split_node(self, node: RadixNode, split_pos: int) -> RadixNode:
        """Split a node at given position, return new intermediate node"""
        new_node = RadixNode(
            children={},
            parent=node.parent,
            key=node.key[:split_pos],
            pages=node.pages[:split_pos],
            ref_count=node.ref_count,
        )

        # Update the original node
        node.parent.children[node.key[0]] = new_node
        node.key = node.key[split_pos:]
        node.pages = node.pages[split_pos:]
        node.parent = new_node
        new_node.children[node.key[0]] = node

        return new_node

    def _update_access_path(self, node: RadixNode) -> None:
        """Update access timestamp along path to root"""
        current_time = int(time.time())
        while node is not None:
            node.last_access_timestamp = current_time
            node = node.parent

    def cache_sequence(
        self, token_ids: List[int], existing_pages: Optional[List[PageInfo]] = None
    ) -> RadixNode:
        """Cache a token sequence, potentially extending existing pages"""
        with self._lock:
            if existing_pages:
                total_cached_tokens = sum(p.token_count for p in existing_pages)
                new_tokens = token_ids[total_cached_tokens:]
                if new_tokens:
                    new_pages = self._allocate_pages(len(new_tokens))
                    pages = existing_pages + new_pages
                else:
                    pages = existing_pages
            else:
                pages = self._allocate_pages(len(token_ids))

            return self._insert_sequence(token_ids, pages)

    def _allocate_pages(self, token_count: int) -> List[PageInfo]:
        """Allocate pages needed for token sequence"""
        pages_needed = (token_count + self.tokens_per_page - 1) // self.tokens_per_page
        page_entries = self.page_pool.acquire_free_pages(pages_needed)

        if not page_entries:
            self._evict_pages(pages_needed)
            page_entries = self.page_pool.acquire_free_pages(pages_needed)
            if not page_entries:
                raise RuntimeError(
                    f"Failed to allocate {pages_needed} pages after eviction"
                )

        pages = []
        tokens_remaining = token_count
        for entry in page_entries:
            tokens_in_page = min(self.tokens_per_page, tokens_remaining)
            pages.append(
                PageInfo(
                    page=entry, token_offset=0, token_count=tokens_in_page, ref_count=1
                )
            )
            tokens_remaining -= tokens_in_page

        return pages

    def _insert_sequence(
        self, token_ids: List[int], pages: List[PageInfo]
    ) -> RadixNode:
        """Insert a sequence into the radix tree"""
        curr_node = self.root
        remaining_tokens = token_ids

        while remaining_tokens:
            first_token = remaining_tokens[0]
            if first_token not in curr_node.children:
                # Create new leaf node
                new_node = RadixNode(
                    children={},
                    parent=curr_node,
                    key=remaining_tokens,
                    pages=pages[len(token_ids) - len(remaining_tokens) :],
                    ref_count=1,
                )
                curr_node.children[first_token] = new_node
                return new_node

            child = curr_node.children[first_token]
            match_len = self._get_match_len(child.key, remaining_tokens)

            if match_len < len(child.key):
                # Split existing node
                split_node = self._split_node(child, match_len)
                if match_len < len(remaining_tokens):
                    # Create new node for remaining tokens
                    new_node = RadixNode(
                        children={},
                        parent=split_node,
                        key=remaining_tokens[match_len:],
                        pages=pages[
                            len(token_ids) - len(remaining_tokens) + match_len :
                        ],
                        ref_count=1,
                    )
                    split_node.children[remaining_tokens[match_len]] = new_node
                    return new_node
                return split_node

            remaining_tokens = remaining_tokens[match_len:]
            curr_node = child

        return curr_node

    def _evict_pages(self, pages_needed: int) -> None:
        """Evict pages using LRU strategy"""
        # Collect all nodes
        nodes = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if not node.children:  # leaf node
                nodes.append(node)

        # Sort by access time
        nodes.sort(key=lambda n: n.last_access_timestamp)

        pages_freed = 0
        for node in nodes:
            if node.ref_count == 0:
                freeable_pages = [p for p in node.pages if p.ref_count == 0]
                self.page_pool.release_pages([p.page for p in freeable_pages])
                pages_freed += len(freeable_pages)

                # Remove node if all pages freed
                if len(freeable_pages) == len(node.pages):
                    del node.parent.children[node.key[0]]

                if pages_freed >= pages_needed:
                    break

    def release_pages(self, pages: List[PageInfo]) -> None:
        """Release references to pages"""
        with self._lock:
            for page in pages:
                page.ref_count -= 1
