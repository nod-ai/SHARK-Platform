# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from torch import Tensor, dtype
import torch.nn.functional as F

from ..kernels import (
    mmt_block_scaled_offset_q4_unsigned,
    mmt_block_scaled_q8,
    mmtfp,
    mmt_super_block_scaled_offset_q4_unsigned,
)

from ..types import (
    BlockScaledLayout,
    BlockScaledI4Layout,
    InferenceTensor,
    PrimitiveTensor,
    QuantizedTensor,
    SuperBlockOffsetScaled_4_6_Layout,
)

from .signatures import *


# Fused FP matmul.
@matmul.override(lhs=Tensor, rhs=Tensor)
def matmul_mmtfp_tensor_tensor(
    lhs: Tensor, rhs: PrimitiveTensor, *, transpose_rhs: bool
):
    # We only accelerate matmuls with transposed RHS set up for inference
    # ... like civilized people.
    if not transpose_rhs:
        return NotImplemented
    if len(lhs.shape) > 3:
        # Only 2d or 3d batch matmuls currently supported.
        return NotImplemented
    return mmtfp(lhs, rhs)


@matmul.override(lhs=Tensor, rhs=PrimitiveTensor)
def matmul_mmtfp_tensor_primitive_tensor(
    lhs: Tensor, rhs: PrimitiveTensor, *, transpose_rhs: bool
):
    return matmul_mmtfp_tensor_tensor(lhs, rhs.as_torch(), transpose_rhs=transpose_rhs)


# Quantized Matmul


@matmul.override(lhs=Tensor, rhs=QuantizedTensor)
def matmul_generic_tensor_block_scaled(
    lhs: Tensor, rhs: QuantizedTensor, *, transpose_rhs: bool
):
    """Generic fallback kernel for block scaled layouts.

    This will unpack and operate generically on planar scales/blocks vs a packed
    struct. This may be fine for certain platforms but there is micro-optimization
    potential if specializing further to the packed layout.
    """
    if not transpose_rhs:
        return NotImplemented
    layout = rhs.layout_type
    if layout is not BlockScaledLayout:
        return NotImplemented
    rhs_unpacked = rhs.unpack()
    assert rhs_unpacked.m is None, "NYI: Q8 block scaled with offset"
    return mmt_block_scaled_q8(lhs, rhs_unpacked.d, rhs_unpacked.qs)


@matmul.override(lhs=Tensor, rhs=QuantizedTensor)
def matmul_generic_tensor_block_scaled_i4(
    lhs: Tensor, rhs: QuantizedTensor, *, transpose_rhs: bool
):
    """Generic fallback kernel for an unsigned, block scaled Q4."""
    if not transpose_rhs:
        return NotImplemented
    layout = rhs.layout_type
    if layout is not BlockScaledI4Layout:
        return NotImplemented
    rhs_unpacked = rhs.unpack()
    assert rhs_unpacked.m is not None, "NYI: Q4 without offset not"
    assert not rhs_unpacked.signed, "NYI: Q4 signed"
    return mmt_block_scaled_offset_q4_unsigned(
        a=lhs, d=rhs_unpacked.d, qs=rhs_unpacked.qs_bit_packed, m=rhs_unpacked.m
    )


@matmul.override(lhs=Tensor, rhs=QuantizedTensor)
def matmul_generic_tensor_super_block_offset_scaled_4_6_i4(
    lhs: Tensor, rhs: QuantizedTensor, *, transpose_rhs: bool
):
    if not transpose_rhs:
        return NotImplemented
    layout = rhs.layout_type
    if layout is not SuperBlockOffsetScaled_4_6_Layout:
        return NotImplemented
    rhs_unpacked = rhs.unpack()
    sb_scales_hi, sb_scales_low = rhs_unpacked.sb_scales_bit_packed
    sb_mins_hi, sb_mins_low = rhs_unpacked.sb_mins_bit_packed
    return mmt_super_block_scaled_offset_q4_unsigned(
        lhs,
        rhs_unpacked.d,
        rhs_unpacked.dmin,
        sb_scales_hi,
        sb_scales_low,
        sb_mins_hi,
        sb_mins_low,
        rhs_unpacked.qs_bit_packed,
    )
