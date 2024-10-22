import array
import logging
import numpy as np

from shortfin import array as sfnp

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


async def to_np(tensor: sfnp.device_array):
    tensor_shape = tensor.shape

    host_tensor = tensor.for_transfer()
    host_tensor.copy_from(tensor)
    await tensor.device
    tensor = host_tensor

    is_float16 = False
    if tensor.dtype == sfnp.float16:
        is_float16 = True
    tensor = tensor.items

    np_array = None
    if isinstance(tensor, array.array):
        dtype = tensor.typecode

        # float16 is not supported by python array, so have to
        # explicitly set dtype for numpy.
        if is_float16:
            dtype = np.float16

        np_array = np.frombuffer(tensor, dtype=dtype)

        # Ensure shape of array matches original tensor shape
        np_array = np_array.reshape(*tensor_shape)

    return np_array


async def dump_array(tensor: sfnp.device_array):
    np_array = await to_np(tensor)
    logger.debug(np_array)


async def fill_array(tensor: sfnp.device_array, fill_value: int | float):
    np_array = await to_np(tensor)
    np_array.fill(fill_value)
    return np_array


def find_mode(arr: np.ndarray, axis=0, keepdims=False):
    """
    Find the mode of an array along a given axis.

    Args:
        arr: The input array.
        axis: The axis along which to find the mode.
        keepdims: If True, the output shape is the same as arr except along the specified axis.

    Returns:
        The mode of the input array.
    """

    def _mode(arr):
        if arr.size == 0:
            return np.nan, 0

        unique, counts = np.unique(arr, return_counts=True)
        max_counts = counts.max()

        mode = unique[counts == max_counts][0]
        return mode, max_counts

    result = np.apply_along_axis(_mode, axis, arr)
    mode_values, mode_count = result[..., 0], result[..., 1]

    if keepdims:
        mode_values = np.expand_dims(mode_values, axis)
        mode_count = np.expand_dims(mode_count, axis)

    return mode_values, mode_count


async def log_tensor_stats(tensor: sfnp.device_array):
    np_array = await to_np(tensor)

    nan_count = np.isnan(np_array).sum()

    # Remove NaN values
    np_array_no_nan = np_array[~np.isnan(np_array)]

    logger.info(f"  NaN count: {nan_count} / {np_array.size}")
    logger.info(f"  Shape: {np_array.shape}, dtype: {np_array.dtype}")

    if len(np_array_no_nan) > 0:
        mode = find_mode(np_array_no_nan)[0]
        logger.info(f"  Min (excluding NaN): {np_array_no_nan.min()}")
        logger.info(f"  Max (excluding NaN): {np_array_no_nan.max()}")
        logger.info(f"  Mean (excluding NaN): {np_array_no_nan.mean()}")
        logger.info(f"  Mode (excluding NaN): {mode}")
        logger.info(
            f"  First 10 elements: {np_array_no_nan.flatten()[:10]}"
            f"  Last 10 elements: {np_array_no_nan.flatten()[-10:]}"
        )
    else:
        logger.warning(f"  All values are NaN")
