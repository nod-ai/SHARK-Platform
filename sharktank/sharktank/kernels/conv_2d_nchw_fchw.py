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
    """Generic convolution on explicitly padded inputs.


    Will be specialized for all values of strides, padding, dilations, and LHS dtype.
    """

    signature = "conv_2d_nchw_fchw(Tensor inputs, Tensor weights, Tensor bias, int[] strides, int[] dilations) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        inputs_desc = ksel.arg_tensor(0)
        weights_desc = ksel.arg_tensor(1)
        bias_desc = ksel.arg_tensor(2)
        strides_desc = ksel.attr_list_int(3)  # Shape [2]
        dilations_desc = ksel.attr_list_int(4)  # Shape [2]

        # unpack
        n, c, h_pad, w_pad = inputs_desc.t.shape
        f, g, k0, k1 = weights_desc.t.shape
        (b,) = bias_desc.t.shape

        strides = strides_desc.v
        dilations = dilations_desc.v

        # check
        torch._check(
            b == f,
            lambda: f"conv_2d_nchw_fchw bias shape should match out_channels shape but got {b} instead of {f}",
        )
        torch._check(
            len(strides) == 2,
            lambda: f"conv_2d_nchw_fchw requires exactly 2 strides; strides: {strides}",
        )
        torch._check(
            len(dilations) == 2,
            lambda: f"conv_2d_nchw_fchw requires exactly 2 dilations; dilations: {dilations}",
        )

        # convolution shape math
        h_out = math.floor((h_pad - dilations[0] * (k0 - 1) - 1) / strides[0] + 1)
        w_out = math.floor((w_pad - dilations[1] * (k1 - 1) - 1) / strides[1] + 1)
        c_desc = ksel.return_new_tensor([n, f, h_out, w_out], dtype=inputs_desc.t.dtype)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        inputs = kb.arg_value(0)
        inputs_tensor_type = RankedTensorType(inputs.type)
        strides = ksel.arg_descs[3].v
        dilations = ksel.arg_descs[4].v
        result_desc = ksel.result_descs[0].t.shape
        H_out = result_desc[2]
        W_out = result_desc[3]

        strides = [str(i) for i in strides]
        dilations = [str(i) for i in dilations]
        H_out = str(H_out)
        W_out = str(W_out)

        dtype_str = str(inputs_tensor_type.element_type)

        template_file = "conv_2d_nchw_fchw.mlir"
        target_function_name = f"sharktank_conv_2d_nchw_fchw_{strides[0]}_{strides[1]}_{dilations[0]}_{dilations[1]}_{dtype_str}"

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            H_out=H_out,
            W_out=W_out,
            strides_H=strides[0],
            strides_W=strides[1],
            dilations_H=dilations[0],
            dilations_W=dilations[1],
            dtype=dtype_str,
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
