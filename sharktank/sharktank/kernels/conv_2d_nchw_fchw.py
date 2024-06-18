# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.kernels.base import *

import math

import torch

__all__ = [
    "conv_2d_nchw_fchw",
]


@CustomOp.register(library=LIBRARY)
class conv_2d_nchw_fchw(CustomOp):
    """Generic block scaled matmul with transposed RHS.

    This corresponds to the BlockScaledLayout and operates on planar `d`
    and `qs` tensors as specified there:

    * `d`: `[N, K // 32, 1]`
    * `qs`: `[N, K // 32, 32]`

    The LHS is expected to be a 3d tensor of shape [B, M, K]. The kernel
    will be specialized for all values of N, K and LHS dtype.
    """

    signature = "conv_2d_nchw_fchw(Tensor inputs, Tensor weights, str c, str d, str e) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        lhs_desc = ksel.arg_tensor(0)
        rhs_desc = ksel.arg_tensor(1)
        strides_desc = ksel.attr_str(2)  # Shape [2]
        padding_desc = ksel.attr_str(3)
        dilations_desc = ksel.attr_str(4)  # Shape [2]

        # a arg
        n, c, h, w = lhs_desc.t.shape
        _, _, k0, k1 = rhs_desc.t.shape
        #lhs_batch, lhs_m, lhs_k = lhs_desc.t.shape

        # d arg
        #rhs_batch, rhs_n, rhs_k = rhs_desc.t.shape
        #torch._check(
        #    rhs_k == lhs_k,
        #    lambda: f"batch_matmul_transpose_b arg 'rhs': Incorrect shape (got {rhs_desc.t.shape})",
        #)

        #strides_count, = strides_desc.t.shape
        #dilations_count, = dilations_desc.t.shape
        strides = strides_desc.v.split(", ")
        strides = [int(i) for i in strides]
        dilations = dilations_desc.v.split(", ")
        dilations = [int(i) for i in dilations]
        padding = padding_desc.v.split(", ")
        padding= [int(i) for i in padding]
        print(strides, padding, dilations)

        h_out = math.floor((h + 2 * padding[0] - dilations[0] * (k0 - 1) - 1) / strides[0] + 1)
        w_out = math.floor((w + 2 * padding[1] - dilations[1] * (k1 - 1) - 1) / strides[1] + 1)
        print(h_out, w_out)

        """torch._check(
            strides_count == 2,
            lambda: f"too many strides"
        )
        torch._check(
            dilations_count == 2,
            lambda: f"too many dilations"
        )"""
        #torch._check(
        #    padding_count == 2,
        #    lambda: f"wrong dimensions of padding"
        #)

        c_desc = ksel.return_new_tensor([n, c, h_out, w_out], dtype=lhs_desc.t.dtype)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        lhs = kb.arg_value(0)
        lhs_tensor_type = RankedTensorType(lhs.type)
        rhs = kb.arg_value(1)
        rhs_tensor_type = RankedTensorType(rhs.type)
        print(kb.symbol_table)
        strides = ksel.arg_descs[2].v
        padding = ksel.arg_descs[3].v
        dilations = ksel.arg_descs[4].v
        strides_str = strides.replace(", ", "x")
        padding_str = padding.replace(", ", "x")
        dilations_str = dilations.replace(", ", "x")

        dtype_str = str(lhs_tensor_type.element_type)

        template_file = "conv_2d_nchw_fchw.mlir"
        target_function_name = (
            f"sharktank_conv_2d_nchw_fchw_{strides_str}_{padding_str}_{dilations_str}_{dtype_str}"
        )

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            strides=strides,
            padding=padding,
            dilations=dilations,
            strides_str=strides_str,
            padding_str=padding_str,
            dilations_str=dilations_str,
            dtype=dtype_str,
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
