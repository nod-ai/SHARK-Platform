# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.kernels.base import *

import torch

__all__ = [
    "batch_matmul_transpose_b",
]


@CustomOp.register(library=LIBRARY)
class batch_matmul_transpose_b(CustomOp):
    """Generic batch matmul.

    * `lhs`: `[B, M, K]`
    * `rhs`: `[B, N, K]`

    The LHS is expected to be a 3d tensor of shape [B, M, K]. The kernel
    will be specialized for all values of M, N, K and LHS dtype.
    """

    signature = "batch_matmul_transpose_b(Tensor lhs, Tensor rhs) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        lhs_desc = ksel.arg_tensor(0)  # Shape [b, m, k]
        rhs_desc = ksel.arg_tensor(1)  # Shape [b, n, k]

        # lhs arg
        lhs_batch, lhs_m, lhs_k = lhs_desc.t.shape

        # rhs arg
        rhs_batch, rhs_n, rhs_k = rhs_desc.t.shape
        torch._check(
            rhs_k == lhs_k and rhs_batch == lhs_batch,
            lambda: f"batch_matmul_transpose_b arg 'rhs': Incorrect shape (got {rhs_desc.t.shape})",
        )

        # Specialize on K, N, BS
        lhs_desc.specialize_dims(-1, -2)
        rhs_desc.specialize_dims(-1, -2)

        # Shape batch..., m, n
        c_desc = ksel.return_new_tensor(
            [lhs_batch, lhs_m, rhs_n], dtype=lhs_desc.t.dtype
        )
        c_desc.specialize_dims(-1, -2)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        lhs = kb.arg_value(0)
        lhs_tensor_type = RankedTensorType(lhs.type)
        rhs = kb.arg_value(1)
        rhs_tensor_type = RankedTensorType(rhs.type)

        b, m, k = lhs_tensor_type.shape
        b, n, k = rhs_tensor_type.shape
        dtype_str = str(lhs_tensor_type.element_type)

        template_file = "batch_matmul_transpose_b.mlir"
        target_function_name = (
            f"sharktank_batch_matmul_transpose_b_{m}_{n}_{k}_{dtype_str}"
        )

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            n=n,
            k=k,
            m=m,
            b=b,
            dtype=dtype_str,
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
