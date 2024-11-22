from typing import Protocol, TypeVar, Callable, Dict, List, Optional, Generic, Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from page_pool import PageInfo
from dataclasses import dataclass
import time
from radix_tree import (
    RadixData,
)  # Protocol for data types that can be stored and managed in a radix tree


T = TypeVar("T")  # The generic data type stored in nodes


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
        return self.path_pages + self.node_pages

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
        ...
