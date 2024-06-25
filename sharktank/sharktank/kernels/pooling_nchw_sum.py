# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.kernels.base import *

import math

import torch

__all__ = [
    "pooling_nchw_sum",
]


@CustomOp.register(library=LIBRARY)
class pooling_nchw_sum(CustomOp):
    """Generic block scaled pooling sum.

    This corresponds to the BlockScaledLayout and operates on planar `d`
    and `qs` tensors as specified there:

    * `d`: `[N, K // 32, 1]`
    * `qs`: `[N, K // 32, 32]`

    """

    signature = "pooling_nchw_sum(Tensor a, int[] weights_size, int[] strides, int[] padding, int[] dilations) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        input_desc = ksel.arg_tensor(0)  # Shape [b, ] m, k
        weights_size_desc = ksel.attr_list_int(1)  # Shape [2]
        strides_desc = ksel.attr_list_int(2)  # Shape [2]
        padding_desc = ksel.attr_list_int(3)  # Shape [2]
        dilations_desc = ksel.attr_list_int(4)  # Shape [2]

        # unpack
        n, c, h, w = input_desc.t.shape

        weights_size = weights_size_desc.v
        strides = strides_desc.v
        padding = padding_desc.v
        dilations = padding_desc.v

        input_desc.specialize_all_dims()

        # pooling shape math
        h_out = math.floor((h + 2 * padding[0] - weights_size[0]) / strides[0] + 1)
        w_out = math.floor((w + 2 * padding[1] - weights_size[1]) / strides[1] + 1)

        c_desc = ksel.return_new_tensor([n, c, h_out, w_out], dtype=input_desc.t.dtype)
        c_desc.specialize_all_dims()

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        input = kb.arg_value(0)
        input_tensor_type = RankedTensorType(input.type)
        input_dim_sizes = "x".join(
            str(input_tensor_type.get_dim_size(i))
            for i in range(input_tensor_type.rank)
        )
        weights = ksel.arg_descs[1].v
        strides = ksel.arg_descs[2].v
        padding = ksel.arg_descs[3].v
        dilations = ksel.arg_descs[4].v
        pooling_nchw_sum = ksel.result_descs[0]
        pooling_nchw_sum_output_shape = pooling_nchw_sum.mlir_type_asm
        input_padding = input_tensor_type.shape
        input_padding[-2] = input_padding[-2] + padding[0] * 2
        input_padding[-1] = input_padding[-1] + padding[1] * 2

        weights = [str(i) for i in weights]
        strides = [str(i) for i in strides]
        padding = [str(i) for i in padding]
        dilations = [str(i) for i in dilations]
        input_padding = [str(i) for i in input_padding]

        dtype_str = str(input_tensor_type.element_type)

        template_file = "pooling_nchw_sum.mlir"
        target_function_name = f"sharktank_pooling_nchw_sum_{input_dim_sizes}_{weights[0]}_{weights[1]}_{strides[0]}_{strides[1]}_{padding[0]}_{padding[1]}_{dilations[0]}_{dilations[1]}_{dtype_str}"

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            input_dim_sizes=input_dim_sizes,
            input_tensor_type=input_tensor_type,
            input_padding_N=input_padding[0],
            input_padding_C=input_padding[1],
            input_padding_H=input_padding[2],
            input_padding_W=input_padding[3],
            pooling_nchw_sum_output_shape=pooling_nchw_sum_output_shape,
            weights_H=weights[0],
            weights_W=weights[1],
            strides_H=strides[0],
            strides_W=strides[1],
            padding_H=padding[0],
            padding_W=padding[1],
            dilations_H=dilations[0],
            dilations_W=dilations[1],
            dtype=dtype_str,
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
