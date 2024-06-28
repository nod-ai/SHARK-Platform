# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Sequence
from ..types.tensors import AnyTensor


def broadcast_dim(
    dim: int, shaped_or_shape: Sequence[Sequence[int] | AnyTensor]
) -> int:
    """Returns the dimension corresponding to `shaped_or_shape[0]`'s dimension `dim` after
    broadcasting `shaped_or_shape`.

    Parameters
    ----------
    args: is a collection of shapes or tensors.

    ```python
    shape1 = [2, 3, 1]
    shape2 = [4, 2, 3, 5]
    d = broadcast_dim(2, shape1, shape2)
    assert d == 3
    ```
    """
    return broadcast_dims([dim], shaped_or_shape)[0]


def broadcast_dims(
    dims: Sequence[int], shaped_or_shape: Sequence[Sequence[int] | AnyTensor]
) -> Sequence[int]:
    """Returns the dimensions corresponding to `shaped_or_shape`s' dimensions after
    broadcasting `shaped_or_shape`.

    Parameters
    ----------
    args: is a collection of shapes or tensors.

    ```python
    shape1 = [2, 3, 1]
    shape2 = [4, 2, 3, 5]
    dims = [2, 2]
    res = broadcast_dims(dims, [shape1, shape2])
    print(res) # [3, 2]
    ```
    """
    assert len(dims) > 0 and len(shaped_or_shape) >= len(dims)
    if hasattr(shaped_or_shape[0], "shape"):
        # Tensors case.
        return broadcast_dims(dims, [tensor.shape for tensor in shaped_or_shape])
    ranks = [len(shape) for shape in shaped_or_shape]
    broadcast_rank = max(ranks)
    return [dim + max(0, broadcast_rank - rank) for dim, rank in zip(dims, ranks)]
