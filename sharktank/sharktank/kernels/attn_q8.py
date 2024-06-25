# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.kernels.base import *

import torch

__all__ = [
    "attn_q8",
]


@CustomOp.register(library=LIBRARY)
class attn_q8(CustomOp):
    """Generic axis scaled attention

    * `q`: `[M, K]
    * `k`: `[N, K]`
    * `v`: `[P, N]`
    * `scale_q`: `[M]
    * `scale_k`: `[N]`
    * `scale_v`: `[P]`
    * `offset_q`: `[M]
    * `offset_k`: `[N]`
    * `offset_v`: `[P]`
    * `attn_mask`: `[M, N]
    * `randoms`: `[M, N]
    * `p`: `[1]
    * `is_causal`: `[1]
    * `scale`: `[1]


    Will be specialized for all values of N, K and LHS dtype.
    """

    signature = "attn_q8(Tensor m0, Tensor m1, Tensor m2, Tensor scale0, Tensor scale1, Tensor scale2, Tensor zp0, Tensor zp1, Tensor zp2, Tensor attn_mask, Tensor randoms, Tensor p, Tensor is_causal, Tensor scale) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        query_desc = ksel.arg_tensor(0)
        key_desc = ksel.arg_tensor(1)
        value_desc = ksel.arg_tensor(2)
        query_s_desc = ksel.arg_tensor(3)
        key_s_desc = ksel.arg_tensor(4)
        value_s_desc = ksel.arg_tensor(5)
        query_zp_desc = ksel.arg_tensor(6)
        key_zp_desc = ksel.arg_tensor(7)
        value_zp_desc = ksel.arg_tensor(8)
        attn_mask_desc = ksel.arg_tensor(9)
        randoms_desc = ksel.arg_tensor(10)
        p_desc = ksel.arg_tensor(11)
        is_causal_desc = ksel.arg_tensor(12)
        scale_desc = ksel.arg_tensor(13)

        # a arg
        query_m, query_k = query_desc.t.shape
        key_n, key_k = key_desc.t.shape
        value_p, value_n = value_desc.t.shape

        # d arg
        # rhs_n, rhs_k, *rest = rhs_desc.t.shape
        # torch._check(
        #    len(rest) == 0 and rhs_k == lhs_k,
        #    lambda: f"scaled_mmt_q8 arg 'rhs': Incorrect shape (got {rhs_desc.t.shape})",
        # )

        query_desc.specialize_all_dims()
        key_desc.specialize_all_dims()
        value_desc.specialize_all_dims()
        query_s_desc.specialize_all_dims()
        key_s_desc.specialize_all_dims()
        value_s_desc.specialize_all_dims()
        query_zp_desc.specialize_all_dims()
        key_zp_desc.specialize_all_dims()
        value_zp_desc.specialize_all_dims()
        attn_mask_desc.specialize_all_dims()
        randoms_desc.specialize_all_dims()
        p_desc.specialize_all_dims()
        is_causal_desc.specialize_all_dims()
        scale_desc.specialize_all_dims()

        # Shape batch..., m, n
        out_desc = ksel.return_new_tensor([query_m, value_p], dtype=torch.float32)
        out_desc.specialize_all_dims()

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        query = kb.arg_value(0)
        query_tensor_type = RankedTensorType(query.type)
        key = kb.arg_value(1)
        key_tensor_type = RankedTensorType(key.type)
        value = kb.arg_value(2)
        value_tensor_type = RankedTensorType(value.type)

        m, k = query_tensor_type.shape
        n, k = key_tensor_type.shape
        p, n = value_tensor_type.shape

        template_file = "attn_q8.mlir"
        target_function_name = f"sharktank_attn_q8"

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            n=n,
            k=k,
            m=m,
            p=p,
            lowp_type="i8",
            a_type="f32",
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
