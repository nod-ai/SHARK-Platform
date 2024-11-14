# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.kernels.base import *

import torch

from iree.turbine.support.ir_imports import (
    ComplexType,
    F16Type,
    RankedTensorType,
    ShapedType,
    Value,
    flow_d,
    tensor_d,
)

from iree.turbine.runtime.op_reg import (
    CustomOp,
    KernelBuilder,
    KernelSelection,
)

__all__ = [
    "bitcast_to_complex",
    "bitcast_to_real",
]


@CustomOp.register(library=LIBRARY)
class bitcast_to_complex(CustomOp):

    signature = "bitcast_to_complex(Tensor q) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        ta = ksel.arg_tensor(0)

        torch._check(ta.t.dtype == torch.float16)
        torch._check(isinstance(ta.t.shape[-1], int))

        new_shape = [i for i in ta.t.shape]
        new_shape[-1] = new_shape[-1] // 2

        ret = ksel.return_new_tensor(new_shape, dtype=torch.complex32)
        specialize_all_known_dims(ta)
        specialize_all_known_dims(ret)

    def eager_execute(self, tensor):
        return torch.view_as_complex(tensor.unflatten(-1, (-1, 2)))

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        t = kb.arg_bindings[0]
        result_desc = ksel.result_descs[0]
        result_shape = [
            d if isinstance(d, int) else RankedTensorType.get_dynamic_size()
            for d in result_desc.t.shape
        ]

        dynamic_dims: list[Value] = []
        _append_dynamic_dims(kb, dynamic_dims, t)

        c64 = ComplexType.get(F16Type.get())
        rtt = RankedTensorType.get(result_shape, c64)
        result = flow_d.TensorBitCastOp(rtt, t, dynamic_dims, dynamic_dims).result
        kb.yield_results(result)


@CustomOp.register(library=LIBRARY)
class bitcast_to_real(CustomOp):

    signature = "bitcast_to_real(Tensor q) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        ta = ksel.arg_tensor(0)

        torch._check(ta.t.dtype == torch.complex32)
        torch._check(isinstance(ta.t.shape[-1], int))

        new_shape = [i for i in ta.t.shape]
        new_shape[-1] = new_shape[-1] * 2

        ret = ksel.return_new_tensor(new_shape, dtype=torch.float16)
        specialize_all_known_dims(ta)
        specialize_all_known_dims(ret)

    def eager_execute(self, tensor):
        return torch.view_as_real(tensor).flatten(-2, -1)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        t = kb.arg_bindings[0]
        result_desc = ksel.result_descs[0]
        result_shape = [
            d if isinstance(d, int) else RankedTensorType.get_dynamic_size()
            for d in result_desc.t.shape
        ]

        dynamic_dims: list[Value] = []
        _append_dynamic_dims(kb, dynamic_dims, t)

        f16 = F16Type.get()
        rtt = RankedTensorType.get(result_shape, f16)
        result = flow_d.TensorBitCastOp(rtt, t, dynamic_dims, dynamic_dims).result
        kb.yield_results(result)


################################################################################
# Emission utilities
################################################################################


def _append_dynamic_dims(kb: KernelBuilder, dynamic_dims: list[Value], tensor: Value):
    rtt = RankedTensorType(tensor.type)
    for i in range(rtt.rank):
        if rtt.is_dynamic_dim(i):
            dynamic_dims.append(tensor_d.dim(tensor, kb.constant_index(i)))
