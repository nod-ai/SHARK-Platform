"""
Generic radix tree implementation.

We should be able to test this by mocking a class that inherits from RadixData, without referencing any other LLM specific classes.

Type checking is set up for radix cache use. Need to ignore those for tests if mocking keys and values.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, TypeVar, Generic
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from page_pool import PageInfo

    KeyItemType = int
    KeyIndexType = int
    ValueItemType = PageInfo

from dataclasses import dataclass
from page_pool import PagePool


class RadixData(Protocol):
    """Protocol defining required operations for data stored in RadixTree nodes"""

    def from_parent(
        self, parent: RadixData, new_values: list[ValueItemType]
    ) -> RadixData:
        """
        Create a new RadixData based on a parent and new values.

        E.g. for a list of kvpages, one might call

        kvlist: List[PageInfo]
        RadixData.from_parent(parent_node, kvlist)

        which might take
        new_values = [p1, p2, p3, p4, p5]
        parent: path_pages = [p1, p2, p3], node_pages = [p4]

        and return
        child: path_pages = [p1, p2, p3, p4], node_pages = [p5]

        (pX are PageInfo objects pointing to pages)

        This should be done purely based off of size and indices but use assertions to ensure consistency, at least during testing.
        """
        ...

    def split_at(
        self, desired_split_point: KeyIndexType
    ) -> Tuple[RadixData, RadixData, KeyIndexType]:
        """
        Split this data at given point, returning (prefix, suffix, actual_split_point).

        actual_split_point <= desired_split_point

        Both split points are key(token)-indices in the radix tree, not value(page) indices.
        """
        ...

    def merge_with_child(self, child: RadixData) -> RadixData:
        """Merge this data with its child, returning combined data"""
        ...

    def on_evict(self) -> None:
        """Called when data is being evicted from tree. Optional cleanup."""
        ...


@dataclass
class RadixNode:
    key: List[int]
    value: RadixData

    children: Dict[int, RadixNode]
    parent: Optional[RadixNode]

    last_access_timestamp: int = 0
    ref_count: int = 0


class RadixTree:
    """A radix tree implementation for caching token sequence data.

    Key = token ids.

    The tree efficiently stores and retrieves cached data for token sequences,
    enabling prefix sharing and fast lookups. It handles memory management through
    reference counting and LRU eviction.
    """

    def __init__(
        self,
    ) -> None:
        """Initialize the radix tree."""
        raise NotImplementedError()

    def match_prefix(self, key: List[KeyItemType]) -> tuple[RadixNode, KeyIndexType]:
        """Find the longest matching prefix for a given key sequence, returning the matched RadixNode and match length.

        Matching will attempt to split nodes, but the split point is determined by RadixData and may not reflect the longest possible match.

        Example - With RadixData that forces splitting at 4-token boundaries:
        ```
        Initial tree:
            [abcd efgh] -> RadixNodePageData(path_pages=[], node_pages=[p1, p2])

        match_prefix([abcd exyz]):
        1. Matches first 5 tokens
        2. RadixData.split_at(5) returns:
            prefix: (path_pages=[], node_pages=[p1])
            suffix: (path_pages=[p1], node_pages=[p2])
            actual_split: 4  # Forces split at block boundary

        Tree after split:
            [abcd] → data(node_pages=[p1])
                └── [efgh] → data(node_pages=[p2])

        Returns: (node containing [abcd], match_length=4)
        ```

        Note:
        - Even if tokens match beyond a block boundary, RadixData.split_at()
          may force splits at specific points
        - The returned match length is determined by RadixData constraints,
          not just by matching tokens
        """
        raise NotImplementedError()

    def set_value(
        self, key: List[KeyItemType], values: list[ValueItemType]
    ) -> RadixNode:
        """
        Ensure that a RadixNode exists in the tree mapping the keys to the value.

        Uses match_prefix to find a place to insert into, then uses RadixData.from_parent to create a value for the new node and inserts a new node.
        ```
        """
        raise NotImplementedError()

    def release_pages(self, pages: List) -> None:
        """Release references to cached pages.

        Decrements reference counts and potentially frees memory if counts reach zero.

        Args:
            pages: List of pages to release
        """
        raise NotImplementedError()

    def _get_match_len(
        self, key1: List[KeyItemType], key2: List[KeyItemType]
    ) -> KeyIndexType:
        """Return length of matching prefix between two keys.

        Args:
            key1: First sequence of tokens
            key2: Second sequence of tokens

        Returns:
            Length of the matching prefix
        """
        raise NotImplementedError()

    def _split_node(
        self, node: RadixNode, desired_split_point: KeyIndexType
    ) -> RadixNode:
        """
        Shrink the current node's key and value as necessary and insert a new node before the current node.

        Example:
            ```pseudocode
            a = RadixNode{
                node_key: "abcd efgh",
                full_key: "abcd efgh",
                node_value: [v_abcd, v_efgh]
                full_value: [v_abcd, v_efgh]
                node
                children: [c1, c2]
            }
            c1 = RadixNode{
                node_key: "ijk"
                node_value: v_ijk
                full_key: "abcd efgh ijk",
                full_value: [v_abcd, v_efgh, v_ijk]
            }
            c2 = RadixNode{
                full_key: "abcd efgh lmn",
                full_value: [v_abcd, v_efgh, v_lmn]
            }

            p1 = tree._split_node(a, desired_split_point = 5) # during attempt to insert abcd eXYZ
            # now:
            p1 = RadixNode{
                node_key == full_key == "abcd"
                node_value == full_value == v_abcd
                children: [a]
            }

            a = RadixNode{
                node_key = 'efgh'
                full_key: "abcd efgh
                node_value = v_efgh
                full_value: [v_abcd, v_efgh]
                children: [c1, c2]
                parent: [p1]
            }
            ```

        """
        raise NotImplementedError()

    def _evict_pages(self, pages_needed: int) -> None:
        """Evict zero reference count leaf pages using LRU strategy until enough pages are free.

        Do not implement

        """
        raise NotImplementedError()
