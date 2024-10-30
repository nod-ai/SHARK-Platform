import logging

import numpy as np

from shortfin import array as sfnp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def debug_dump_array(tensor: sfnp.device_array) -> None:
    """Dump the contents of a device array to the debug log.

    Args:
        tensor (sfnp.device_array): The device array to dump.
    """
    np_array = np.array(tensor)
    logger.debug(np_array)


def debug_fill_array(tensor: sfnp.device_array, fill_value: int | float) -> np.ndarray:
    """Fill a device array with a given value and return the resulting numpy array.

    Args:
        tensor (sfnp.device_array): The device array to fill.
        fill_value (int | float): The value to fill the array with.

    Returns:
        np.ndarray: The filled numpy array.
    """
    np_array = np.array(tensor)
    np_array.fill(fill_value)
    return np_array


def _find_mode(
    arr: np.ndarray, axis=0, keepdims=False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the mode of an array along a given axis.

    Args:
        arr: The input array.
        axis: The axis along which to find the mode.
        keepdims: If True, the output shape is the same as arr except along the specified axis.

    Returns:
        tuple: A tuple containing the mode values and the count of the mode values.
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


def debug_log_tensor_stats(tensor: sfnp.device_array) -> None:
    """Log statistics about a device array to the debug log.

    The following statistics are logged:
    - NaN count
    - Shape, dtype
    - Min, max, mean, mode (excluding NaN values)
    - First 10 elements
    - Last 10 elements

    Args:
        tensor (sfnp.device_array): The device array to log statistics for.
    """

    np_array = np.array(tensor)

    nan_count = np.isnan(np_array).sum()

    # Remove NaN values
    np_array_no_nan = np_array[~np.isnan(np_array)]

    logger.debug(f"NaN count: {nan_count} / {np_array.size}")
    logger.debug(f"Shape: {np_array.shape}, dtype: {np_array.dtype}")

    if len(np_array_no_nan) > 0:
        mode = _find_mode(np_array_no_nan)[0]
        logger.debug(f"Min (excluding NaN): {np_array_no_nan.min()}")
        logger.debug(f"Max (excluding NaN): {np_array_no_nan.max()}")
        logger.debug(f"Mean (excluding NaN): {np_array_no_nan.mean()}")
        logger.debug(f"Mode (excluding NaN): {mode}")
        logger.debug(f"First 10 elements: {np_array_no_nan.flatten()[:10]}")
        logger.debug(f"Last 10 elements: {np_array_no_nan.flatten()[-10:]}")
    else:
        logger.warning(f"All values are NaN")
