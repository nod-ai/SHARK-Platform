import logging
import time
import asyncio
from typing import Callable, Any

logger = logging.getLogger(__name__)

# measurement helper
def measure(fn):

    """
    Decorator log test start and end time of a function
    :param fn: Function to decorate
    :return: Decorated function
    Example:
    >>> @timed
    >>> def test_fn():
    >>>     time.sleep(1)
    >>> test_fn()
    """

    def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        ret = fn(*args, **kwargs)
        duration_str = get_duration_str(start)
        logger.info(f"Completed {fn.__qualname__} in {duration_str}")
        return ret

    async def wrapped_fn_async(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        ret = await fn(*args, **kwargs)
        duration_str = get_duration_str(start)
        logger.info(f"Completed {fn.__qualname__} in {duration_str}")
        if fn.__qualname__ == "ClientGenerateBatchProcess.run":
            sps_str = get_samples_per_second(start, *args)
            logger.info(f"SAMPLES PER SECOND = {sps_str}")
        return ret

    if asyncio.iscoroutinefunction(fn):
        return wrapped_fn_async
    else:
        return wrapped_fn


def get_samples_per_second(start, *args: Any) -> str:
    duration = time.time() - start
    bs = args[0].gen_req.num_output_images
    sps = str(float(bs) / duration)
    return sps


def get_duration_str(start: float) -> str:
    """Get human readable duration string from start time"""
    duration = time.time() - start
    duration_str = f"{round(duration * 1e3)}ms"
    return duration_str
