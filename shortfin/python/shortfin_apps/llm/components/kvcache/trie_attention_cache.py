from typing import Dict, Set, List, Tuple
from .page_pool import PageInfo
import heapq
from dataclasses import dataclass
import time
from .base_attention_cache import BasePagedAttentionCache
import math


@dataclass
class TrieNode:
    """Node of the block trie for paged attention cache."""

    tokens: List[int]
    page: PageInfo
    num_matched: int = 0
    children: Dict[int, "TrieNode"] = None
    _parent: "TrieNode" = None

    def __post_init__(self):
        if self.children is None:
            self.children = {}

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, val: "TrieNode"):
        old_parent = self._parent
        if old_parent is not None:
            old_parent.children.pop(hash(tuple(self.tokens)))
        if val is not None:
            val.children[hash(tuple(self.tokens))] = self
        self._parent = val

    # nodes are uniquely identified by their memory address
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class TriePagedAttentionCache(BasePagedAttentionCache):
    """
    Trie-based paged attention cache that manages PageInfo objects.
    Implements prefix sharing through a trie structure where each node
    represents a page of tokens.
    """

    def __init__(self, page_pool, tokens_per_page, enable_prefix_sharing=True):
        super().__init__(page_pool, tokens_per_page)
        self.enable_prefix_sharing = enable_prefix_sharing
        dummy_page = PageInfo(
            index=-1, pool=self.page_pool, token_offset=0, token_count=0
        )
        self.root = TrieNode(tokens=[], page=dummy_page)
        self.leaves: Set[TrieNode] = set()

    def list_equal(self, list1: List[int], list2: List[int]) -> bool:
        """Compare two lists for equality."""
        if len(list1) != len(list2):
            return False
        return all(a == b for a, b in zip(list1, list2))

    def acquire_pages_for_tokens(
        self,
        tokens: List[int],
        extra_token_slots: int = 1,
    ) -> Tuple[List[PageInfo], int]:
        if not self.enable_prefix_sharing:
            return super().acquire_pages_for_tokens(tokens, extra_token_slots)

        tokens = list(tokens)
        pages = []
        curr_node = self.root
        n_cached_tokens = curr_node.num_matched

        # Try to match existing pages
        while n_cached_tokens + self.tokens_per_page <= len(tokens):
            curr_tokens = tokens[
                n_cached_tokens : n_cached_tokens + self.tokens_per_page
            ]
            key = hash(tuple(curr_tokens))

            if key not in curr_node.children:
                break

            child = curr_node.children[key]
            if not self.list_equal(curr_tokens, child.tokens):
                break

            # Match found - increment reference count
            child.page.read_ref_count += 1
            pages.append(child.page)
            curr_node = child
            n_cached_tokens += self.tokens_per_page

        # Allocate new pages for remaining tokens plus extra slots
        remaining_tokens = len(tokens) - n_cached_tokens + extra_token_slots
        if remaining_tokens > 0:
            pages_needed = math.ceil(remaining_tokens / self.tokens_per_page)
            new_pages = self.page_pool.acquire_free_pages(pages_needed)

            # If allocation failed, try eviction and retry
            if new_pages is None:
                self.evict_pages(pages_needed)  # Call existing evict_pages method
                new_pages = self.page_pool.acquire_free_pages(pages_needed)

            if new_pages:
                pages.extend(new_pages)
            else:
                # If we still can't get pages after eviction, return None
                for page in pages:  # Release any pages we previously matched
                    page.read_ref_count -= 1
                return None, 0

        return pages, n_cached_tokens

    def publish_pages(
        self, tokens: List[int], pages: List[PageInfo], model_variant: str = "default"
    ) -> None:
        """
        Publish pages to make them available for prefix sharing.
        """
        if not self.enable_prefix_sharing:
            return

        tokens = list(tokens)  # Create a copy to ensure we have a list
        curr_node = self.root
        num_matched = curr_node.num_matched

        # Remove from leaves if it was a leaf
        if len(curr_node.children) == 0 and curr_node.parent is not None:
            self.leaves.remove(curr_node)

        # Add new nodes to trie
        for i, page in enumerate(pages):
            if num_matched + self.tokens_per_page > len(tokens):
                break

            curr_tokens = tokens[num_matched : num_matched + self.tokens_per_page]
            hash_key = hash(tuple(curr_tokens))

            parent = curr_node
            if hash_key in parent.children:
                child = parent.children[hash_key]
                if self.list_equal(curr_tokens, child.tokens):
                    curr_node = child
                    # Update page info
                    page.read_ref_count -= 1  # Remove our reference
                    if page.read_ref_count == 0:
                        self.page_pool.release_pages([page])
                    continue

            # Create new node
            curr_node = TrieNode(
                tokens=curr_tokens,
                page=page,
                num_matched=num_matched + self.tokens_per_page,
            )
            curr_node.parent = parent

            num_matched += self.tokens_per_page

        # Add to leaves if it's a new leaf
        if curr_node.parent is not None and len(curr_node.children) == 0:
            self.leaves.add(curr_node)

    def release_pages(self, tokens: List[int], pages: List[PageInfo]) -> None:
        """
        Release pages and decrement their reference counts.
        When reference count reaches zero, the page becomes eligible for eviction.
        """
        for page in pages:
            page.read_ref_count -= 1
            if page.read_ref_count == 0:
                self.page_pool.release_pages([page])

    def evict_pages(self, max_pages: int) -> int:
        """
        Evict up to max_pages pages from the cache, starting with least recently used.
        Returns number of pages actually evicted.
        """
        if not self.enable_prefix_sharing:
            return 0

        def __remove_leaf(leaves, evicted_pages):
            _, leaf = heapq.heappop(leaves)
            evicted_pages.append(leaf.page)
            parent = leaf.parent
            leaf.parent = None
            self.leaves.remove(leaf)
            return parent

        def __add_leaf(leaves, parent):
            self.leaves.add(parent)
            if parent.page.read_ref_count == 0:
                # Only add to eviction candidates if not referenced
                heapq.heappush(leaves, (time.time(), parent))

        evicted_pages = []
        leaf_heap = []

        # Initialize heap with unreferenced leaves
        for leaf in self.leaves:
            if leaf.page.read_ref_count == 0:
                # Using current time as priority - could be replaced with proper LRU tracking
                heapq.heappush(leaf_heap, (time.time(), leaf))

        while leaf_heap and len(evicted_pages) < max_pages:
            parent = __remove_leaf(leaf_heap, evicted_pages)
            if parent is None or parent.parent is None:
                # Skip root
                continue
            if len(parent.children) == 0:
                __add_leaf(leaf_heap, parent)

        if evicted_pages:
            self.page_pool.release_pages(evicted_pages)

        return len(evicted_pages)
