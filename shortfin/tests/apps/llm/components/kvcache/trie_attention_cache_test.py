import pytest
from typing import List, Tuple
import shortfin as sf
import shortfin.array as sfnp
from unittest.mock import Mock, MagicMock
import threading
import time
from dataclasses import dataclass

from shortfin_apps.llm.components.kvcache.trie_attention_cache import (
    TriePagedAttentionCache,
)
from shortfin_apps.llm.components.kvcache.base_attention_cache import (
    CacheAllocationFailure,
)
from shortfin_apps.llm.components.kvcache.page_pool import (
    PagePool,
    PageInfo,
    PagePoolConfig,
)


# Test constants
TEST_PAGE_SIZE = 16  # Tokens per page
TEST_POOL_CAPACITY = 10


@dataclass
class TokenSequence:
    """Helper class for test parameterization"""

    tokens: List[int]
    description: str
    expected_pages: int
    expected_cached: int = 0

    def __str__(self):
        return self.description


class MockScopedDevice:
    """A proper mock for ScopedDevice that implements required interface"""

    def __init__(self):
        self._mock = Mock(spec=sf.ScopedDevice)
        # Add any necessary attributes/methods the real ScopedDevice has
        self._mock.device_id = 0
        self._mock.device_type = "CPU"

    def __repr__(self):
        return f"MockScopedDevice(device_id={self._mock.device_id})"


@pytest.fixture
def mock_device_array():
    """Create mock device array with proper interface implementation"""

    class MockDeviceArray:
        def __init__(self):
            self.shape = None
            self.dtype = None

        def view(self, *args):
            return Mock()

        def copy_from(self, src):
            pass

    return MockDeviceArray()


@pytest.fixture
def mock_device():
    """Create properly structured mock device"""
    return MockScopedDevice()


@pytest.fixture
def page_pool(mock_device, mock_device_array):
    """Create PagePool with properly structured mock components"""
    # Mock the device array creation
    original_for_device = sf.array.device_array.for_device

    def mock_for_device(device, shape, dtype):
        mock_array = mock_device_array
        mock_array.shape = shape
        mock_array.dtype = dtype
        return mock_array

    sf.array.device_array.for_device = mock_for_device

    try:
        config = PagePoolConfig(
            dtype=sfnp.float16,
            alloc_page_count=TEST_POOL_CAPACITY,
            paged_kv_block_size_elements=128,
        )

        pool = PagePool(devices=[mock_device], config=config)
        pool.page_tables = [mock_device_array]
        return pool
    finally:
        # Restore original function
        sf.array.device_array.for_device = original_for_device


@pytest.fixture
def trie_cache(page_pool):
    """Create TriePagedAttentionCache instance"""
    return TriePagedAttentionCache(page_pool=page_pool, tokens_per_page=TEST_PAGE_SIZE)


@pytest.fixture
def published_sequence(trie_cache):
    """Helper fixture that returns a function to publish token sequences"""

    def _publish_sequence(tokens: List[int]) -> None:
        alloc = trie_cache.acquire_pages_for_tokens(tokens, extra_token_slots=0)
        alloc.publish_pages_for_tokens(alloc.tokens)
        alloc.release_pages()

    return _publish_sequence


def print_tree_state(cache, prefix=""):
    """Helper function to print current tree state in a readable format"""
    if not hasattr(cache, "root"):
        print(f"{prefix}Unable to access trie structure")
        return

    def node_info(node):
        token_str = f"tokens={list(node.tokens) if node.tokens else 'root'}"
        return f"{token_str}, ref_count={node.ref_count}, page_index={node.page.index}"

    def print_node(node, depth=0):
        indent = "  " * depth
        print(f"{prefix}{indent}- {node_info(node)}")
        if node.children:
            for child in node.children.values():
                print_node(child, depth + 1)

    print(f"{prefix}Tree state:")
    print_node(cache.root)


# Test sequences for parameterization
basic_sequences = [
    TokenSequence(tokens=[], description="empty_sequence", expected_pages=0),
    TokenSequence(
        tokens=list(range(TEST_PAGE_SIZE // 2)),
        description="partial_page",
        expected_pages=1,
    ),
    TokenSequence(
        tokens=list(range(TEST_PAGE_SIZE)), description="exact_page", expected_pages=1
    ),
    TokenSequence(
        tokens=list(range(TEST_PAGE_SIZE + 1)),
        description="overflow_page",
        expected_pages=2,
    ),
    TokenSequence(
        tokens=list(range(TEST_PAGE_SIZE * 2)),
        description="multiple_pages",
        expected_pages=2,
    ),
]

reuse_sequences = [
    (list(range(TEST_PAGE_SIZE)), list(range(TEST_PAGE_SIZE)), "exact_match", 1, 1),
    (
        list(range(TEST_PAGE_SIZE * 2)),
        list(range(TEST_PAGE_SIZE * 2)),
        "multi_page_match",
        2,
        2,
    ),
    (
        list(range(TEST_PAGE_SIZE * 2)),
        list(range(TEST_PAGE_SIZE)) + list(range(100, 100 + TEST_PAGE_SIZE)),
        "prefix_match",
        2,
        1,
    ),
    (
        list(range(TEST_PAGE_SIZE)),
        list(range(50, 50 + TEST_PAGE_SIZE)),
        "no_match",
        1,
        0,
    ),
]


@pytest.mark.parametrize("seq", basic_sequences)
def test_basic_allocation(trie_cache, seq):
    """Test basic page allocation without reuse"""
    allocation = trie_cache.acquire_pages_for_tokens(seq.tokens, extra_token_slots=0)
    assert len(allocation.pages) == seq.expected_pages
    assert allocation.number_of_published_pages == 0
    assert (
        len(allocation.pages) - allocation.number_of_published_pages
        == seq.expected_pages
    )
    allocation.publish_pages_for_tokens(allocation.tokens)
    allocation.release_pages()


@pytest.mark.parametrize(
    "initial_tokens,reuse_tokens,description,total_pages,expected_cached",
    reuse_sequences,
)
def test_page_reuse(
    trie_cache,
    published_sequence,
    initial_tokens,
    reuse_tokens,
    description,
    total_pages,
    expected_cached,
):
    """Test page reuse scenarios"""
    # Publish initial sequence
    published_sequence(initial_tokens)

    # Try to reuse
    allocation = trie_cache.acquire_pages_for_tokens(reuse_tokens, extra_token_slots=0)
    assert len(allocation.pages) == total_pages
    assert allocation.number_of_published_pages == expected_cached
    assert (
        len(allocation.pages) - allocation.number_of_published_pages
        == total_pages - expected_cached
    )
    allocation.publish_pages_for_tokens(allocation.tokens)
    allocation.release_pages()


@pytest.fixture
def filled_cache(trie_cache, published_sequence):
    """Fixture that fills cache with numbered sequences"""
    sequences = []
    for i in range(TEST_POOL_CAPACITY):
        tokens = list(range(i * 100, i * 100 + TEST_PAGE_SIZE))
        published_sequence(tokens)
        sequences.append(tokens)
    return sequences


@pytest.mark.parametrize(
    "access_count", [1, TEST_POOL_CAPACITY // 2, TEST_POOL_CAPACITY - 1]
)
def test_lru_eviction(trie_cache, access_count):
    """Test LRU eviction with different access patterns"""
    print(f"\nStarting test_lru_eviction with access_count={access_count}")

    # Create mix of published and unpublished sequences
    keep_published = 3  # Number of sequences to keep published
    sequences = []

    # First add some sequences we'll keep published
    print("\nPublishing sequences to keep active:")
    for i in range(keep_published):
        tokens = list(range(i * 100, i * 100 + TEST_PAGE_SIZE))
        alloc = trie_cache.acquire_pages_for_tokens(tokens, extra_token_slots=0)
        alloc.publish_pages_for_tokens(alloc.tokens[:TEST_PAGE_SIZE])
        sequences.append(tokens)
        print(f"Published sequence {i} (keeping active)")
        print_tree_state(trie_cache, "  ")

    # Then add sequences we'll publish but release (evictable)
    print("\nAdding releasable sequences:")
    for i in range(keep_published, TEST_POOL_CAPACITY):
        tokens = list(range(i * 100, i * 100 + TEST_PAGE_SIZE))
        alloc = trie_cache.acquire_pages_for_tokens(tokens, extra_token_slots=0)
        alloc.publish_pages_for_tokens(alloc.tokens[:TEST_PAGE_SIZE])
        alloc.release_pages()  # These can be evicted
        sequences.append(tokens)
        print(f"Added releasable sequence {i}")
        print_tree_state(trie_cache, "  ")

    print("\nCache state before accessing sequences:")
    print_tree_state(trie_cache, "  ")

    # Access some sequences to update their LRU status
    print(f"\nAccessing {access_count} sequences to update LRU order:")
    for i in range(access_count):
        print(f"\nAccessing sequence {i}:")
        alloc = trie_cache.acquire_pages_for_tokens(sequences[i], extra_token_slots=0)
        print_tree_state(trie_cache, "  ")
        alloc.release_pages()
        print(f"After releasing allocation {i}:")
        print_tree_state(trie_cache, "  ")

    print("\nCache state before attempting new allocation:")
    print_tree_state(trie_cache, "  ")
    print("\nAvailable pages in pool:", len(trie_cache.page_pool.available_pages))

    # Try to allocate new sequence - should evict least recently used unpublished sequence
    new_tokens = list(range(1000, 1000 + TEST_PAGE_SIZE))
    print(f"\nAttempting to allocate new sequence: {new_tokens}")
    new_alloc = trie_cache.acquire_pages_for_tokens(new_tokens, extra_token_slots=0)
    print("\nNew allocation succeeded:")
    print("\nCache state after new allocation:")
    print_tree_state(trie_cache, "  ")
    new_alloc.release_pages()

    # Verify recently accessed sequences AND published sequences weren't evicted
    print("\nVerifying preserved sequences:")
    for i in range(max(access_count, keep_published)):
        print(f"\nChecking sequence {i}:")
        recheck = trie_cache.acquire_pages_for_tokens(sequences[i], extra_token_slots=0)
        cached_pages = recheck.number_of_published_pages
        print(f"- Cached pages found: {cached_pages}")
        assert (
            cached_pages == 1
        ), f"Sequence {i} was evicted but should have been preserved"
        recheck.release_pages()


@pytest.mark.parametrize("publish_steps", [1, 2, 3])
def test_progressive_publish(trie_cache, publish_steps):
    """Test publishing pages progressively"""
    print(f"\nStarting test_progressive_publish with publish_steps={publish_steps}")

    tokens = tuple(range(TEST_PAGE_SIZE * 3))  # Three pages
    print(f"\nInitial tokens: {tokens}")
    print(f"Tokens per page: {TEST_PAGE_SIZE}")
    print(
        f"Expected total pages: {len(tokens) // TEST_PAGE_SIZE + (1 if len(tokens) % TEST_PAGE_SIZE else 0)}"
    )

    print("\nInitial cache state:")
    print_tree_state(trie_cache)

    print("\nAcquiring initial allocation...")
    alloc = trie_cache.acquire_pages_for_tokens(tokens)
    print(f"Initial allocation pages: {[p.index for p in alloc.pages]}")
    print("\nCache state after initial allocation:")
    print_tree_state(trie_cache)

    for step in range(1, publish_steps + 1):
        print(f"\n--- Step {step} of {publish_steps} ---")

        # Publish next page
        print(f"Publishing up to page {step}")
        # Replace publishing with tokens
        alloc.publish_pages_for_tokens(alloc.tokens[: (step) * TEST_PAGE_SIZE])
        print("\nCache state after publish:")
        print_tree_state(trie_cache)

        # Verify reuse up to published point
        reuse_tokens = tokens[: (step) * TEST_PAGE_SIZE]
        print(f"\nAttempting to reuse tokens: {reuse_tokens}")
        print(f"Expected cached pages: {step}")

        reuse_alloc = trie_cache.acquire_pages_for_tokens(reuse_tokens)
        print(f"Reuse allocation total pages: {len(reuse_alloc.pages)}")
        print(f"Reuse allocation cached pages: {reuse_alloc.number_of_published_pages}")

        print("\nCache state after reuse attempt:")
        print_tree_state(trie_cache)

        try:
            assert reuse_alloc.number_of_published_pages == step
        except AssertionError:
            print("\nASSERTION FAILED!")
            print(
                f"Expected {step} cached pages but got {reuse_alloc.number_of_published_pages}"
            )
            raise

        reuse_alloc.release_pages()
        print("\nCache state after releasing reuse allocation:")
        print_tree_state(trie_cache)

    alloc.release_pages()
    print("\nFinal cache state after releasing initial allocation:")
    print_tree_state(trie_cache)


@pytest.mark.parametrize("ref_count", [1, 2, 5])
def test_reference_counting(trie_cache, ref_count):
    """Test reference counting with different counts"""
    tokens = list(range(TEST_PAGE_SIZE))
    allocations = []

    # Create initial allocation and publish
    first_alloc = trie_cache.acquire_pages_for_tokens(tokens, extra_token_slots=0)
    # Replace publishing with tokens
    first_alloc.publish_pages_for_tokens(first_alloc.tokens)
    allocations.append(first_alloc)
    print("\nInitial allocation created")
    print_tree_state(trie_cache, "  ")

    # Create additional references
    for i in range(ref_count - 1):
        alloc = trie_cache.acquire_pages_for_tokens(tokens, extra_token_slots=0)
        allocations.append(alloc)
        print(f"\nCreated reference {i+1}")
        print_tree_state(trie_cache, "  ")

    # Fill remaining cache
    remaining = TEST_POOL_CAPACITY - 1
    fill_allocations = []
    for i in range(remaining):
        fill_tokens = list(
            range(100 + i * TEST_PAGE_SIZE, 100 + (i + 1) * TEST_PAGE_SIZE)
        )
        alloc = trie_cache.acquire_pages_for_tokens(fill_tokens, extra_token_slots=0)
        alloc.publish_pages_for_tokens(alloc.tokens[:TEST_PAGE_SIZE])
        fill_allocations.append(alloc)
        print(f"\nFilled cache slot {i+1}/{remaining}")
        print_tree_state(trie_cache, "  ")

    print("\nAttempting allocation that should fail...")
    try:
        new_tokens = list(range(1000, 1000 + TEST_PAGE_SIZE))
        new_alloc = trie_cache.acquire_pages_for_tokens(new_tokens, extra_token_slots=0)
        print("ERROR: Allocation succeeded when it should have failed!")
        print("\nPost-allocation state:")
        print_tree_state(trie_cache, "  ")
        new_alloc.release_pages()
        pytest.fail("Expected CacheAllocationFailure was not raised")
    except CacheAllocationFailure:
        print("Success: CacheAllocationFailure raised as expected")

    # Cleanup
    print("\nCleaning up allocations...")
    for alloc in allocations + fill_allocations:
        alloc.release_pages()
