# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Implementations for op variants that are fully quantized.
"""

from typing import Optional

import torch
import warnings


from torch import Tensor

from sharktank import kernels

from types import NoneType

from ..types import (
    AnyTensor,
    QuantizedTensor,
    PlanarQuantizedTensor,
    TensorScaledLayout,
)
from ..utils import debugging

from ..types.tensors import unbox_tensor
from .signatures import (
    IntOrSequenceInt,
    scaled_dot_product_attention,
    elementwise,
)


def flash_attention(q, k, v, a):
    q = unbox_tensor(q)
    k = unbox_tensor(k)
    v = unbox_tensor(v)
    return kernels.flash_attention(q, k, v)


scaled_dot_product_attention.override(AnyTensor, AnyTensor, AnyTensor, NoneType)(flash_attention)
