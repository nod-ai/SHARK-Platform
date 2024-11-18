import pytest
import logging
from shortfin_apps.llm.components.kvcache.page_pool import PagePool, PagePoolConfig
import shortfin as sf
import shortfin.host
import shortfin.array as sfnp
import shortfin.amdgpu

logger = logging.getLogger(__name__)


@pytest.fixture(
    params=[("cpu", sf.host.CPUSystemBuilder), ("gpu", sf.amdgpu.SystemBuilder)]
)
def setup_system(request):
    system_type, builder_class = request.param
    logger.info(f"=== Setting up {system_type.upper()} system ===")
    sc = builder_class()
    lsys = sc.create_system()
    fiber = lsys.create_fiber()
    devices = fiber.devices_dict.values()
    yield system_type, lsys, devices
    lsys.shutdown()


@pytest.fixture
def setup_pool(setup_system):
    system_type, _, devices = setup_system
    logger.info(f"Creating PagePool for {system_type.upper()} system")
    pool = PagePool(
        devices=devices,
        config=PagePoolConfig(
            alloc_page_count=256,
            dtype=sfnp.float16,
            paged_kv_block_size_elements=393216,
        ),
    )
    return system_type, pool


def test_page_acquisition(setup_pool):
    system_type, pool = setup_pool
    logger.info(
        f"=== Running page acquisition test on {system_type.upper()} system ==="
    )
    page0 = pool.acquire_free_pages(1)
    assert page0 is not None, f"Failed to acquire a free page on {system_type} system"
    logger.info(f"Successfully acquired page on {system_type.upper()} system")


def test_page_copy(setup_pool):
    system_type, pool = setup_pool
    logger.info(f"=== Running page copy test on {system_type.upper()} system ===")
    (page0,) = pool.acquire_free_pages(1)
    page1 = pool.copy_page(page0)
    assert page1 is not None, f"Failed to copy a page on {system_type} system"
    assert (
        page0 != page1
    ), f"Copied page should be different from original on {system_type} system"
    logger.info(f"Successfully copied page on {system_type.upper()} system")


@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging format to include timestamp and level"""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        force=True,
    )


# Add more tests as needed

if __name__ == "__main__":
    pytest.main([__file__])
