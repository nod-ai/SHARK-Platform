# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import logging
from dataclasses import astuple, dataclass
from enum import Enum
from typing import Optional

from iree.compiler import ir  # type: ignore

from iree.compiler.dialects import iree_gpu  # type: ignore


class CommonTypes:
    def __init__(self, ctx: ir.Context):
        assert ctx
        self.i1 = ir.IntegerType.get_signless(1, ctx)
        self.i8 = ir.IntegerType.get_signless(8, ctx)
        self.i16 = ir.IntegerType.get_signless(16, ctx)
        self.i32 = ir.IntegerType.get_signless(32, ctx)

        self.f8E4M3FNUZ = ir.Float8E4M3FNUZType.get(ctx)
        self.f8E5M2FNUZ = ir.Float8E5M2FNUZType.get(ctx)
        self.f16 = ir.F16Type.get(ctx)
        self.f32 = ir.F32Type.get(ctx)

        self.bf16 = ir.BF16Type.get(ctx)


class TunerContext:
    def __init__(self, mlir_ctx: ir.Context, logger: logging.Logger):
        self.mlir_ctx: ir.Context = mlir_ctx
        self.logger: logging.Logger = logger
        self.type: CommonTypes = CommonTypes(mlir_ctx)


class DispatchKind(Enum):
    conv = 1
    mmt = 2
    contraction = 3
    batch_mmt = 4
    batch_matmul = 5
    broadcast_rhs_mmt = 6


@dataclass
class ShapedType:
    shape: list[int]
    element_type: ir.IntegerType | ir.FloatType

    def rank(self) -> int:
        return len(self.shape)

    @property
    def bitwidth(self) -> int:
        return self.element_type.width

    def __str__(self) -> str:
        dim_to_str = lambda dim: str(dim) if dim != -1 else "?"
        return "x".join(map(dim_to_str, self.shape)) + "x" + str(self.element_type)


@dataclass
class MatmulSize:
    M: int
    N: int
    K: int
    B: int = 1


@dataclass
class ProblemSize:
    matmul_size: MatmulSize
    lhs_type: ShapedType
    rhs_type: ShapedType
    res_type: ShapedType
    dispatch_kind: DispatchKind

    @property
    def MNK(self) -> tuple[int, int, int]:
        return (self.matmul_size.M, self.matmul_size.N, self.matmul_size.K)


def get_compatible_mfma_intrinsics(
    problem_size: ProblemSize,
    mma_intrinsics: list[iree_gpu.MMAIntrinsic],
) -> list[iree_gpu.MMAIntrinsic]:
    def is_comptible(mma_intrinsic: iree_gpu.MMAIntrinsic) -> bool:
        mma_attr = iree_gpu.MMAIntrinsicAttr.get(mma_intrinsic).mma
        a_type, b_type, c_type = mma_attr.abc_element_types
        if problem_size.res_type.element_type != c_type:
            return False
        if problem_size.dispatch_kind != DispatchKind.batch_matmul:
            if (
                problem_size.lhs_type.element_type != a_type
                or problem_size.rhs_type.element_type != b_type
            ):
                return False
        return True

    return list(filter(is_comptible, mma_intrinsics))


class ReorderWorkgroupsStrategy(Enum):
    NONE = 0
    SWIZZLE = 1
    TRANSPOSE = 2

    def __str__(self) -> str:
        return self.name.title()


@dataclass
class Configuration:
    subgroup_size: int
    workgroup_size: list[int]
    intrinsic: iree_gpu.MMAAttr
    tile_sizes: list[int]
    subgroup_m_count: int
    subgroup_n_count: int
    gpu_pipeline_options: iree_gpu.PipelineOptionsAttr
    waves_per_eu: int


def get_pipeline_config(configuration: Configuration) -> str:
    extra_config = ""
    pipeline_options = configuration.gpu_pipeline_options
    if pipeline_options != iree_gpu.PipelineOptionsAttr.get():
        extra_config += f", gpu_pipeline_options = {pipeline_options}"

    if configuration.waves_per_eu != 2:
        extra_config += f', llvm_func_attrs = {{"amdgpu-waves-per-eu" = "{configuration.waves_per_eu}"}}'
    return extra_config


def read_input_mlir(filename: str) -> list[str]:
    with open(filename, "r") as f:
        return f.readlines()


@dataclass
class ConvDimInfo:
    n: int
    oh: int
    ow: int
    oc: int
    fh: int
    fw: int
    ic: int

    @staticmethod
    def from_rhs_res(rhs_shaped_type: ShapedType, res_shaped_type: ShapedType):
        n, oh, ow, oc = res_shaped_type.shape
        fh, fw, ic, _ = rhs_shaped_type.shape
        return ConvDimInfo(n, oh, ow, oc, fh, fw, ic)

    @staticmethod
    def from_problem_size(problem_size: ProblemSize):
        return ConvDimInfo.from_rhs_res(problem_size.rhs_type, problem_size.res_type)


@dataclass
class MLIRTransformation:
    """Transformation of MLIR context"""

    template: list[str]
    modified: str
    embeddable: str
