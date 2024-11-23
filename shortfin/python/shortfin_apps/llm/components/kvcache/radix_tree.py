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

    def create_child(self: RadixData, new_values: list[ValueItemType]) -> RadixData:
        """
        Create a new RadixData based on a parent and new values.

        E.g. for a list of kvpages, one might call

        kvlist: List[PageInfo]
        parent_node.create_child(kvlist)

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


class RadixNodePageData(RadixData):
    """
    Container for pages associated with a radix tree node.

    This class maintains two distinct collections of pages:
    - node_pages: Pages specifically associated with this individual node
    - path_pages: Collection of pages from the root up to (but not including)
                 this node's pages
    """

    # page data management

    def __init__(self, path_pages, node_pages, tokens_per_page):
        """
        Initialize a RadixNodePageData instance.

        Args:
            path_pages: Collection of pages associated with the path from root
                        up to (but not including) this node
            node_pages: Collection of pages associated only with this specific node
        """
        self.node_pages = node_pages
        self.path_pages = path_pages
        self.tokens_per_page = tokens_per_page

    @property
    def full_pages(self) -> list[PageInfo]:
        """
        Returns all the pages associated with this Node.
        """
        # if profiling reveals that this takes a lot of time,
        # consider moving to an approach where all pages in the same path
        # share the same page-list, and use an index mark the boundary between
        # path and node pages
        return self.path_pages + self.node_pages

    def create_child(self: RadixData, new_values: list[PageInfo]) -> RadixData:
        return RadixNodePageData(
            path_pages=self.full_pages,
            node_pages=new_values,
            tokens_per_page=self.tokens_per_page,
        )

    # radix tree interface functions

    def split_at(
        self, desired_split_point: int
    ) -> Tuple["RadixNodePageData", "RadixNodePageData", int]:
        """
        Split this node at a given point. Has to be split at whole-page boundaries.
        """
        key_begin = len(self.path_pages) * self.tokens_per_page
        key_end = len(self.full_pages) * self.tokens_per_page
        assert (
            desired_split_point != key_begin
        ), "We shouldn't be splitting at the beginning of a page because we should just be adding another RadixNode before this one."
        assert (
            desired_split_point != key_end
        ), "We shouldn't be splitting at the end of a page because we should just add another RadixNode after this one."
        # the split point should be within this node's associated pages
        # we use gt and lt and not ge / le because if the desired_split_point is page aligned, we should have never splitted
        assert (
            len(self.path_pages) * self.tokens_per_page < desired_split_point
        ), "Split point should not come before this node."
        assert desired_split_point < len(
            self.full_pages
        ), "Split point should not comes after this node."

        page_split_point = desired_split_point // self.tokens_per_page
        actual_split_point = page_split_point * self.tokens_per_page

        first_half = self.node_pages[:page_split_point]
        second_half = self.node_pages[page_split_point:]

        new_node = RadixNodePageData(path_pages=self.path_pages, node_pages=first_half)
        new_child = RadixNodePageData(
            path_pages=self.path_pages + first_half, node_pages=second_half
        )
        return new_node, new_child, actual_split_point

    def merge_with_child(self, child: "RadixNodePageData") -> "RadixNodePageData":
        """
        Merge this node with its only child.

        This should never be called on a node with more than one child
        """
        # Child's path_pages should equal our path_pages + our node_pages
        assert child.path_pages == self.path_pages + self.node_pages

        return RadixNodePageData(
            path_pages=self.path_pages, node_pages=self.node_pages + child.node_pages
        )

    def on_evict(self):
        """
        Handles cleanup when this node is evicted from the radix tree.

        Note: This method will only be called on leaf nodes, as the radix tree
        only performs eviction on nodes without children.
        """
        # evict self.node_pages only
        # TODO: remove checks when stable
        assert len(self.node_pages) > 0
        pool = self.node_pages[0].pool
        for p in self.node_pages:
            assert p.pool == pool
        pool.release_pages(self.node_pages)


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
        tokens_per_page,
    ) -> None:
        """Initialize the radix tree."""
        root_data = RadixNodePageData(
            path_pages=[], node_pages=[], tokens_per_page=tokens_per_page
        )
        self.root = RadixNode(key=[], value=root_data, children={}, parent=None)

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
        current_node = self.root
        match_length = 0

    def match_helper(
        self, key: List[KeyItemType], current_node: RadixNode
    ) -> tuple[RadixNode, KeyIndexType]:
        """Helper function for match_prefix."""
        best_match = current_node
        best_match_len = self._get_match_len(key, current_node.key)

        parent_key = current_node.parent.key if current_node.parent else []

        # found match in this node; splitting may be necessary
        if best_match_len < len(current_node.key) and best_match_len > parent_key:
            best_match = self._split_node(current_node, best_match_len)
            assert self._get_match_len(key, best_match.key) == best_match_len
            return best_match, len(best_match.key)

        assert best_match_len == current_node.key

        for c in best_match.children[key[len(current_node.key)]]:
            match, match_len = self.match_helper(key, c)
            if match_len > best_match_len:
                best_match = match
                best_match_len = match_len

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
        match_len = 0
        for k1, k2 in zip(key1, key2):
            if k1 != k2:
                break
            match_len += 1
        return match_len

    def _split_node(
        self, node: RadixNode, desired_split_point: KeyIndexType
    ) -> RadixNode:
        """
        Shrink the current node's key and value as necessary so we can insert a new node as close as possible to desired_split_point.

        Returns: a node to insert after. If we can't split the node, node.parent is returned.

        Returns the new node.

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
        parent_key_length = len(node.parent.key) if node.parent else 0
        # Attempting to split at beginning or end of node would just return node.parent or node
        if desired_split_point == parent_key_length:
            return node.parent
        elif desired_split_point == len(node.key):
            return node

        # Split the node's value to get prefix and suffix
        prefix_data, suffix_data, actual_split_point = node.value.split_at(
            desired_split_point
        )

        # no new node needed if split point is at the beginning of node
        if actual_split_point == parent_key_length:
            return node.parent

        # Create a new parent node with the prefix data
        new_parent = RadixNode(
            key=node.key[:actual_split_point],
            value=prefix_data,
            children={node},
            parent=node.parent,
        )

        node.value = suffix_data
        node.parent = new_parent

        return new_parent

    def _evict_pages(self, pages_needed: int) -> None:
        """Evict zero reference count leaf pages using LRU strategy until enough pages are free.

        Do not implement

        """
        raise NotImplementedError()
