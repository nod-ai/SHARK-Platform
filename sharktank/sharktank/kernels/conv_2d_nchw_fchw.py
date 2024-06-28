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
    """Generic convolution


    Will be specialized for all values of strides, padding, dilations, and LHS dtype.
    """

    signature = "conv_2d_nchw_fchw(Tensor inputs_pad, Tensor weights, Tensor bias, int[] strides, int[] padding, int[] dilations) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        inputs_pad_desc = ksel.arg_tensor(0)
        weights_desc = ksel.arg_tensor(1)
        bias_desc = ksel.arg_tensor(2)
        strides_desc = ksel.attr_list_int(3)  # Shape [2]
        padding_desc = ksel.attr_list_int(4)  # Shape [2]
        dilations_desc = ksel.attr_list_int(5)  # Shape [2]

        # unpack
        n, c, h_pad, w_pad = inputs_pad_desc.t.shape
        f, g, k0, k1 = weights_desc.t.shape
        (b,) = bias_desc.t.shape

        strides = strides_desc.v
        dilations = dilations_desc.v
        padding = padding_desc.v

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
        torch._check(
            len(padding) == 2,
            lambda: f"conv_2d_nchw_fchw requires exactly 2 padding; padding: {padding}",
        )

        # subtract padding shape
        h_out = h_pad - padding[0] - padding[0]
        w_out = w_pad - padding[1] - padding[1]
        c_desc = ksel.return_new_tensor(
            [n, f, h_out, w_out], dtype=inputs_pad_desc.t.dtype
        )

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        inputs_pad = kb.arg_value(0)
        inputs_pad_tensor_type = RankedTensorType(inputs_pad.type)
        strides = ksel.arg_descs[3].v
        padding = ksel.arg_descs[4].v
        dilations = ksel.arg_descs[5].v
        result_desc = ksel.result_descs[0].t.shape
        H_input_pad = result_desc[2] + padding[0] * 2
        W_input_pad = result_desc[3] + padding[1] * 2

        strides = [str(i) for i in strides]
        padding = [str(i) for i in padding]
        dilations = [str(i) for i in dilations]
        H_input_pad = str(H_input_pad)
        W_input_pad = str(W_input_pad)

        dtype_str = str(inputs_pad_tensor_type.element_type)

        template_file = "conv_2d_nchw_fchw.mlir"
        target_function_name = f"sharktank_conv_2d_nchw_fchw_{strides[0]}_{strides[1]}_{padding[0]}_{padding[1]}_{dilations[0]}_{dilations[1]}_{dtype_str}"

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            H_input_pad=H_input_pad,
            W_input_pad=W_input_pad,
            strides_H=strides[0],
            strides_W=strides[1],
            padding_H=padding[0],
            padding_W=padding[1],
            dilations_H=dilations[0],
            dilations_W=dilations[1],
            dtype=dtype_str,
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
