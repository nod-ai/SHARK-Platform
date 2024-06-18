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

    signature = "conv_2d_nchw_fchw(Tensor inputs, Tensor weights, int[] strides, int[] padding, int[] dilations) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        lhs_desc = ksel.arg_tensor(0)
        rhs_desc = ksel.arg_tensor(1)
        strides_desc = ksel.attr_list_int(2)  # Shape [2]
        padding_desc = ksel.attr_list_int(3) # Shape [2]
        dilations_desc = ksel.attr_list_int(4)  # Shape [2]

        # unpack
        n, c, h, w = lhs_desc.t.shape
        f, g, k0, k1 = rhs_desc.t.shape

        strides = strides_desc.v
        dilations = dilations_desc.v
        padding = padding_desc.v

        # check
        torch._check(
            c == f and f == g,
            lambda: f"conf_2d_nchw_fchw arg 'weights': Incorrect shape (got {weights_desc.t.shape})",
        )
        torch._check(
            len(strides) == 2,
            lambda: f"conv_2d_nchw_fchw requires exactly 2 strides; strides: {strides}"
        )
        torch._check(
            len(dilations) == 2,
            lambda: f"conv_2d_nchw_fchw requires exactly 2 dilations; dilations: {dilations}"
        )
        torch._check(
            len(padding) == 2,
            lambda: f"conv_2d_nchw_fchw requires exactly 2 padding; padding: {padding}"
        )

        # convolution shape math
        h_out = math.floor((h + 2 * padding[0] - dilations[0] * (k0 - 1) - 1) / strides[0] + 1)
        w_out = math.floor((w + 2 * padding[1] - dilations[1] * (k1 - 1) - 1) / strides[1] + 1)

        c_desc = ksel.return_new_tensor([n, c, h_out, w_out], dtype=lhs_desc.t.dtype)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        lhs = kb.arg_value(0)
        lhs_tensor_type = RankedTensorType(lhs.type)
        rhs = kb.arg_value(1)
        rhs_tensor_type = RankedTensorType(rhs.type)
        strides = ksel.arg_descs[2].v
        padding = ksel.arg_descs[3].v
        dilations = ksel.arg_descs[4].v
        strides = [str(i) for i in strides]
        padding = [str(i) for i in padding]
        dilations = [str(i) for i in dilations]
        strides_list_str = ", ".join(strides)
        strides_str = "x".join(strides)
        padding_list_str = ", ".join(padding)
        padding_str = "x".join(padding)
        dilations_list_str = ", ".join(dilations)
        dilations_str = "x".join(dilations)

        dtype_str = str(lhs_tensor_type.element_type)

        template_file = "conv_2d_nchw_fchw.mlir"
        target_function_name = (
            f"sharktank_conv_2d_nchw_fchw_{strides_str}_{padding_str}_{dilations_str}_{dtype_str}"
        )

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            strides=strides_list_str,
            padding=padding_list_str,
            dilations=dilations_list_str,
            strides_str=strides_str,
            padding_str=padding_str,
            dilations_str=dilations_str,
            dtype=dtype_str,
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
