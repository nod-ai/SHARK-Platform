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


class TunerContext:
    def __init__(self, mlir_ctx: ir.Context, logger: logging.Logger):
        self.mlir_ctx = mlir_ctx
        self.logger = logger


class DispatchKind(Enum):
    conv = 1
    mmt = 2
    contraction = 3
    batch_mmt = 4
    batch_matmul = 5
    broadcast_rhs_mmt = 6


class ElementType(Enum):
    i8 = 1
    i32 = 2
    f8 = 3
    f16 = 4
    f32 = 5

    @property
    def bitwidth(self) -> int:
        match self:
            case ElementType.i8 | ElementType.f8:
                return 8
            case ElementType.f16:
                return 16
            case ElementType.i32 | ElementType.f32:
                return 32
            case _:
                assert False, "unhandled case"

    def __str__(self) -> str:
        return self.name


@dataclass
class ShapedType:
    shape: list[int]
    element_type: ElementType

    def rank(self) -> int:
        return len(self.shape)

    @property
    def bitwidth(self) -> int:
        return self.element_type.bitwidth

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


@dataclass
class MfmaIntrinsic:
    output_type: ElementType
    m: int
    n: int
    k: int
    input_type: ElementType

    def __str__(self) -> str:
        input = str(self.input_type).upper()
        output = str(self.output_type).upper()
        return f"MFMA_{output}_{self.m}x{self.n}x{self.k}_{input}"

    @staticmethod
    def mfma_f32_16x16x16_f16():
        return MfmaIntrinsic(ElementType.f32, 16, 16, 16, ElementType.f16)

    @staticmethod
    def mfma_f32_32x32x8_f16():
        return MfmaIntrinsic(ElementType.f32, 32, 32, 8, ElementType.f16)

    @staticmethod
    def mfma_i32_16x16x32_i8():
        return MfmaIntrinsic(ElementType.i32, 16, 16, 32, ElementType.i8)

    @staticmethod
    def mfma_i32_32x32x16_i8():
        return MfmaIntrinsic(ElementType.i32, 32, 32, 16, ElementType.i8)

    @staticmethod
    def all():
        return [
            MfmaIntrinsic.mfma_f32_16x16x16_f16(),
            MfmaIntrinsic.mfma_f32_32x32x8_f16(),
            MfmaIntrinsic.mfma_i32_16x16x32_i8(),
            MfmaIntrinsic.mfma_i32_32x32x16_i8(),
        ]


def get_compatible_mfma_intrinsics(problem_size: ProblemSize) -> list[MfmaIntrinsic]:
    def is_compatible(intrinsic: MfmaIntrinsic) -> bool:
        if problem_size.res_type.element_type != intrinsic.output_type:
            return False
        if problem_size.dispatch_kind != DispatchKind.batch_matmul:
            if problem_size.lhs_type.element_type != intrinsic.input_type:
                return False
            if problem_size.rhs_type.element_type != intrinsic.input_type:
                return False
        return True

    return list(filter(is_compatible, MfmaIntrinsic.all()))


class ReorderWorkgroupsStrategy(Enum):
    NONE = 0
    SWIZZLE = 1
    TRANSPOSE = 2

    def __str__(self) -> str:
        return self.name.title()


@dataclass
class GpuPipelineOptions:
    """Represents the `iree_gpu.pipeline_options` attribute"""

    prefetch_shared_memory: Optional[bool] = None
    no_reduce_shared_memory_bank_conflicts: Optional[bool] = None
    reorder_workgroups_strategy: Optional[ReorderWorkgroupsStrategy] = None

    def all_default(self) -> bool:
        return all(x is None for x in astuple(self))

    def __str__(self) -> str:
        options: list[str] = []
        if self.prefetch_shared_memory is not None:
            options.append(
                f"prefetch_shared_memory = {str(self.prefetch_shared_memory).lower()}"
            )
        if self.no_reduce_shared_memory_bank_conflicts is not None:
            options.append(
                f"no_reduce_shared_memory_bank_conflicts = {str(self.no_reduce_shared_memory_bank_conflicts).lower()}"
            )
        if self.reorder_workgroups_strategy is not None:
            options.append(
                f"reorder_workgroups_strategy = {self.reorder_workgroups_strategy}"
            )

        return f"#iree_gpu.pipeline_options<{', '.join(options)}>"


@dataclass
class Configuration:
    subgroup_size: int
    workgroup_size: list[int]
    intrinsic: MfmaIntrinsic
    tile_sizes: list[int]
    subgroup_m_count: int
    subgroup_n_count: int
    gpu_pipeline_options: GpuPipelineOptions
    waves_per_eu: int


def get_pipeline_config(configuration: Configuration) -> str:
    extra_config = ""
    if not configuration.gpu_pipeline_options.all_default():
        extra_config += f", gpu_pipeline_options = {configuration.gpu_pipeline_options}"
    if configuration.waves_per_eu != 2:
        extra_config += f', llvm_func_attrs = {{"amdgpu-waves-per-eu" = "{configuration.waves_per_eu}"}}'
    return extra_config


class MlirRegex(Enum):
    ssa_value = r"%[a-zA-Z0-9-_]+"
    tensor_type = r"tensor<(([0-9]+x)+((f|i)[0-9]+))>"

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def dps_ins_two_args() -> str:
        return rf"ins\({MlirRegex.ssa_value}, {MlirRegex.ssa_value} : (?P<LHS>{MlirRegex.tensor_type}), (?P<RHS>{MlirRegex.tensor_type})\)"

    @staticmethod
    def dps_outs_one_arg() -> str:
        return rf"outs\({MlirRegex.ssa_value} : (?P<RES>{MlirRegex.tensor_type})\)"


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


def parse_tensor_type(tensor_type: str) -> ShapedType:
    shape_match = re.search(str(MlirRegex.tensor_type), tensor_type)
    assert shape_match

    shape_str = shape_match.group(1)
    dims_and_elem = shape_str.split("x")
    dims = [int(x) for x in dims_and_elem[:-1]]
    elem = dims_and_elem[-1]
    str_to_elem_ty = {x.name: x for x in ElementType}
    return ShapedType(dims, str_to_elem_ty[elem])


@dataclass
class MLIRTransformation:
    """Transformation of MLIR context"""

    template: list[str]
    modified: str
    embeddable: str
