# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.kernels.base import *

import torch

__all__ = [
    "mmt_scaled_q8",
]


@CustomOp.register(library=LIBRARY)
class mmt_scaled_q8(CustomOp):
    """Generic block scaled matmul with transposed RHS.

    This corresponds to the BlockScaledLayout and operates on planar `d`
    and `qs` tensors as specified there:

    * `d`: `[N, K // 32, 1]`
    * `qs`: `[N, K // 32, 32]`

    The LHS is expected to be a 3d tensor of shape [B, M, K]. The kernel
    will be specialized for all values of N, K and LHS dtype.
    """

    signature = "mmt_scaled_q8(Tensor lhs, Tensor rhs, Tensor scale0, Tensor scale1, Tensor zp0, Tensor zp1) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        lhs_desc = ksel.arg_tensor(0)  # Shape [b, ] m, k
        scale0_desc = ksel.arg_tensor(2)  # Shape [N, K // BLOCK_SIZE, BLOCK_SIZE]
        scale1_desc = ksel.arg_tensor(3)  # Shape [N, K // BLOCK_SIZE, BLOCK_SIZE]
        rhs_desc = ksel.arg_tensor(1)  # Shape [N, K // BLOCK_SIZE, 1]
        zp0_desc = ksel.arg_tensor(4)
        zp1_desc = ksel.arg_tensor(5)

        # a arg
        lhs_m, lhs_k = lhs_desc.t.shape

        # d arg
        rhs_n, rhs_k, *rest = rhs_desc.t.shape
        torch._check(
            len(rest) == 0 and rhs_k == lhs_k,
            lambda: f"scaled_mmt_q8 arg 'rhs': Incorrect shape (got {rhs_desc.t.shape})",
        )

        # Specialize on K, N, BS
        lhs_desc.specialize_all_dims()
        scale0_desc.specialize_all_dims()
        scale1_desc.specialize_all_dims()
        rhs_desc.specialize_all_dims()
        zp0_desc.specialize_all_dims()
        zp1_desc.specialize_all_dims()

        # Shape batch..., m, n
        c_desc = ksel.return_new_tensor([lhs_m, rhs_n], dtype=torch.float32)
        c_desc.specialize_all_dims()

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        lhs = kb.arg_value(0)
        lhs_tensor_type = RankedTensorType(lhs.type)
        rhs = kb.arg_value(1)
        rhs_tensor_type = RankedTensorType(rhs.type)
        scale0 = kb.arg_value(2)
        scale0_tensor_type = RankedTensorType(scale0.type)
        scale1 = kb.arg_value(3)
        scale1_tensor_type = RankedTensorType(scale1.type)
        zp0 = kb.arg_value(4)
        zp0_tensor_type = RankedTensorType(zp0.type)
        zp1 = kb.arg_value(5)
        zp1_tensor_type = RankedTensorType(zp1.type)

        m, k = lhs_tensor_type.shape
        n, k = rhs_tensor_type.shape
        lhs_type_str = str(lhs_tensor_type.element_type)

        template_file = "mmt_scaled_q8.mlir"
        target_function_name = f"mmt_scaled_q8"

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            n=n,
            k=k,
            m=m,
            lowp_type="i8",
            a_type="f32",
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
