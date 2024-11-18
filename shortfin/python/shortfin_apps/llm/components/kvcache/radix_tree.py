from __future__ import annotations
from typing import List, Dict, Optional, Tuple, TypeVar, Generic
from dataclasses import dataclass

T = TypeVar("T")  # Generic type for page data


@dataclass
class RadixNode(Generic[T]):
    """A node in the radix tree that tracks cached pages of data.

    Each node represents a sequence of tokens and maintains references to the pages
    containing the cached data for those tokens. The node structure allows for
    efficient prefix matching and sharing of cached data.

    Attributes:
        children: Mapping of first token to child nodes
        parent: Reference to parent node, None for root
        key: Token sequence this node represents
        pages: List of page data associated with this token sequence
        last_access_timestamp: Unix timestamp of last access
        ref_count: Number of active references to this node

    Example:
        ```python
        # Create a leaf node for token sequence [5, 2, 8]
        node = RadixNode(
            children={},
            parent=parent_node,
            key=[5, 2, 8],
            pages=[page1, page2],
            ref_count=1
        )

        # Access timestamp is automatically updated when node is accessed
        assert node.last_access_timestamp > 0

        # When done with node, decrement reference count
        node.ref_count -= 1
        ```
    """

    children: Dict[int, RadixNode[T]]
    parent: Optional[RadixNode[T]]
    key: List[int]
    pages: List[T]
    last_access_timestamp: int = 0
    ref_count: int = 0


class RadixTree(Generic[T]):
    """A radix tree implementation for caching token sequence data.

    The tree efficiently stores and retrieves cached data for token sequences,
    enabling prefix sharing and fast lookups. It handles memory management through
    reference counting and LRU eviction.

    Example:
        ```python
        # Initialize tree with a page pool and 16 tokens per page
        tree = RadixTree(page_pool=my_pool, tokens_per_page=16)

        # Cache a sequence of tokens with their associated data
        token_ids = [1, 5, 8, 2]
        node = tree.cache_sequence(token_ids)

        # Later, find cached data for a prefix
        pages, match_node = tree.match_prefix([1, 5, 8])
        assert len(pages) > 0

        # When done with the cached data, release it
        tree.release_pages(pages)
        ```
    """

    def __init__(
        self, *, page_pool: Any, tokens_per_page: int, disable: bool = False
    ) -> None:
        """Initialize the radix tree.

        Args:
            page_pool: Pool that manages the underlying page allocations
            tokens_per_page: Number of tokens that can be stored in each page
            disable: If True, disables caching (useful for testing)

        Example:
            ```python
            tree = RadixTree(
                page_pool=PagePool(...),
                tokens_per_page=16,
                disable=False
            )
            ```
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """Reset the tree to initial empty state.

        Releases all cached pages and resets the tree to contain only a root node.

        Example:
            ```python
            tree = RadixTree(...)

            # Cache some sequences
            tree.cache_sequence([1, 2, 3])
            tree.cache_sequence([4, 5, 6])

            # Reset tree to clean state
            tree.reset()

            # Tree is now empty except for root node
            pages, _ = tree.match_prefix([1, 2, 3])
            assert len(pages) == 0
            ```
        """
        raise NotImplementedError()

    def match_prefix(self, token_ids: List[int]) -> Tuple[List[T], RadixNode[T]]:
        """Find the longest matching prefix and return its cached pages.

        Args:
            token_ids: Sequence of tokens to match against

        Returns:
            Tuple containing:
            - List of cached pages for the matching prefix
            - The node containing the last matched token

        Example:
            ```python
            # Cache a sequence
            tree.cache_sequence([1, 2, 3, 4, 5])

            # Match a prefix
            pages, node = tree.match_prefix([1, 2, 3])

            # pages contains cached data for tokens [1, 2, 3]
            assert len(pages) > 0

            # node represents the position after [1, 2, 3]
            assert node.key == [1, 2, 3]

            # Don't forget to release when done
            tree.release_pages(pages)
            ```
        """
        raise NotImplementedError()

    def cache_sequence(
        self, token_ids: List[int], existing_pages: Optional[List[T]] = None
    ) -> RadixNode[T]:
        """Cache a token sequence, potentially extending existing cached pages.

        Args:
            token_ids: Complete sequence of tokens to cache
            existing_pages: Optional list of already cached pages to extend

        Returns:
            Node containing the cached sequence

        Example:
            ```python
            # Cache initial sequence
            node1 = tree.cache_sequence([1, 2, 3])

            # Match prefix and extend with new tokens
            pages, _ = tree.match_prefix([1, 2, 3])
            node2 = tree.cache_sequence([1, 2, 3, 4, 5], existing_pages=pages)

            # New node contains extended sequence
            assert node2.key == [1, 2, 3, 4, 5]

            # Release pages when done
            tree.release_pages(pages)
            ```
        """
        raise NotImplementedError()

    def release_pages(self, pages: List[T]) -> None:
        """Release references to cached pages.

        Decrements reference counts and potentially frees memory if counts reach zero.

        Args:
            pages: List of pages to release

        Example:
            ```python
            # Get cached pages
            pages, _ = tree.match_prefix([1, 2, 3])

            # Use the pages...

            # Release when done
            tree.release_pages(pages)
            ```
        """
        raise NotImplementedError()

    def _get_match_len(self, key1: List[int], key2: List[int]) -> int:
        """Return length of matching prefix between two keys.

        Args:
            key1: First sequence of tokens
            key2: Second sequence of tokens

        Returns:
            Length of the matching prefix

        Example:
            ```python
            # Internal use for finding split points
            length = tree._get_match_len([1, 2, 3, 4], [1, 2, 5, 6])
            assert length == 2  # Matches [1, 2]
            ```
        """
        raise NotImplementedError()

    def _split_node(self, node: RadixNode[T], split_pos: int) -> RadixNode[T]:
        """Split a node at the given position.

        Args:
            node: Node to split
            split_pos: Position in the node's key where split should occur

        Returns:
            New intermediate node created by the split

        Example:
            ```python
            # Internal use during insertion
            # If we have a node with key [1, 2, 3, 4] and need to
            # insert [1, 2, 5, 6], we first split at position 2:

            old_node.key == [1, 2, 3, 4]
            new_node = tree._split_node(old_node, 2)

            assert new_node.key == [1, 2]
            assert old_node.key == [3, 4]
            assert old_node.parent == new_node
            ```
        """
        raise NotImplementedError()

    def _evict_pages(self, pages_needed: int) -> None:
        """Evict pages using LRU strategy until enough pages are free.

        Args:
            pages_needed: Number of pages that need to be freed

        Example:
            ```python
            # Internal use when cache is full
            # If we need 5 pages and cache is full:
            tree._evict_pages(5)

            # After eviction, at least 5 pages should be available
            pages = page_pool.acquire_free_pages(5)
            assert pages is not None
            ```
        """
        raise NotImplementedError()
