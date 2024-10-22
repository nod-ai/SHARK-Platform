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
from abc import abstractmethod

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
    mmt4d = 7


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


@dataclass
class MfmaIntrinsic:
    output_type: ir.IntegerType | ir.FloatType
    m: int
    n: int
    k: int
    input_type: ir.IntegerType | ir.FloatType

    def __str__(self) -> str:
        input = str(self.input_type).upper()
        output = str(self.output_type).upper()
        return f"MFMA_{output}_{self.m}x{self.n}x{self.k}_{input}"

    @staticmethod
    def mfma_f32_16x16x16_f16():
        f16 = ir.F16Type.get()
        f32 = ir.F32Type.get()
        return MfmaIntrinsic(f32, 16, 16, 16, f16)

    @staticmethod
    def mfma_f32_32x32x8_f16():
        f16 = ir.F16Type.get()
        f32 = ir.F32Type.get()
        return MfmaIntrinsic(f32, 32, 32, 8, f16)

    @staticmethod
    def mfma_i32_16x16x32_i8():
        i32 = ir.IntegerType.get_signless(32)
        i8 = ir.IntegerType.get_signless(8)
        return MfmaIntrinsic(i32, 16, 16, 32, i8)

    @staticmethod
    def mfma_i32_32x32x16_i8():
        i32 = ir.IntegerType.get_signless(32)
        i8 = ir.IntegerType.get_signless(8)
        return MfmaIntrinsic(i32, 32, 32, 16, i8)

    @staticmethod
    def all():
        return [
            MfmaIntrinsic.mfma_f32_16x16x16_f16(),
            MfmaIntrinsic.mfma_f32_32x32x8_f16(),
            MfmaIntrinsic.mfma_i32_16x16x32_i8(),
            MfmaIntrinsic.mfma_i32_32x32x16_i8(),
        ]


def get_compatible_mfma_intrinsics(
    problem_size: ProblemSize,
    mma_intrinsics: list[iree_gpu.MMAIntrinsic],
) -> list[MfmaIntrinsic]:
    available_mma_intrinsics = [str(mma) for mma in mma_intrinsics]

    def is_compatible(intrinsic: MfmaIntrinsic) -> bool:
        if problem_size.res_type.element_type != intrinsic.output_type:
            return False
        if problem_size.dispatch_kind != DispatchKind.batch_matmul:
            if problem_size.lhs_type.element_type != intrinsic.input_type:
                return False
            if problem_size.rhs_type.element_type != intrinsic.input_type:
                return False

        if str(intrinsic) not in available_mma_intrinsics:
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
class BaseConfiguration:
    config_logger = logging.getLogger("tune")
    tile_sizes: list[int]

    @abstractmethod
    def get_pipeline_config(self) -> str:
        pass

    @abstractmethod
    def get_intrinsic_config(self) -> str:
        pass

    def get_base_mlir_config(self, tile_sizes: list[int]) -> str:
        tile_sizes_str = ", ".join(map(str, tile_sizes))
        return f"""
    %config = transform.param.constant #iree_codegen.compilation_info<
        lowering_config = #iree_codegen.lowering_config<tile_sizes = [[{tile_sizes_str}]]>,
    """

    @abstractmethod
    def get_mlir_config(self, tile_sizes: list[int]) -> str:
        base_config = self.get_base_mlir_config(tile_sizes)
        full_config = f"""
        translation_info = #iree_codegen.translation_info<NONE>
        > -> !transform.any_param
    """

        return base_config + full_config

    @abstractmethod
    def apply_configuration(self, template: list[str], tile_sizes: list[int]) -> str:
        pass


@dataclass
class LLVMGPUConfiguration(BaseConfiguration):
    subgroup_size: int
    workgroup_size: list[int]
    intrinsic: MfmaIntrinsic
    subgroup_m_count: int
    subgroup_n_count: int
    gpu_pipeline_options: GpuPipelineOptions
    waves_per_eu: int

    def get_pipeline_config(self) -> str:
        extra_config = ""
        if not self.gpu_pipeline_options.all_default():
            extra_config += f", gpu_pipeline_options = {self.gpu_pipeline_options}"
        if self.waves_per_eu != 2:
            extra_config += (
                f', llvm_func_attrs = {{"amdgpu-waves-per-eu" = "{self.waves_per_eu}"}}'
            )

        return extra_config

    def get_intrinsic_config(self) -> str:
        return str(self.intrinsic)

    def get_mlir_config(self, tile_sizes: list[int]) -> str:
        base_config = self.get_base_mlir_config(tile_sizes)
        wg_x, wg_y, wg_z = self.workgroup_size

        backend_config = f"""
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
            workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {self.subgroup_size},
            {{mma_schedule = #iree_gpu.mma_schedule<
                intrinsic = #iree_gpu.mma_layout<{self.get_intrinsic_config()}>,
                subgroup_m_count = {self.subgroup_m_count}, subgroup_n_count = {self.subgroup_n_count}>
            {self.get_pipeline_config()}}}>
        > -> !transform.any_param
    """

        return base_config + backend_config

    def apply_configuration(self, template: list[str], tile_sizes: list[int]) -> str:
        self.config_logger.info(f"Applying: {self}")

        expr0 = re.compile(
            r"<intrinsic = #iree_gpu\.mma_layout<(.+)>, subgroup_m_count = ([0-9]+), subgroup_n_count = ([0-9]+)>"
        )
        expr1 = re.compile(
            r"LLVMGPUVectorDistribute workgroup_size = \[.+\] subgroup_size = ([0-9]+),"
        )
        expr2 = re.compile(r"tile_sizes = \[\[([0-9]+)(, ([0-9]+))+\]\]")
        expr3 = re.compile(
            r"gpu_pipeline_options = #iree_gpu\.pipeline_options<([^>]*)>"
        )
        expr4 = re.compile(r"\"amdgpu-waves-per-eu\" = \"([0-9])\"")

        repl0 = f"<intrinsic = #iree_gpu.mma_layout<{self.intrinsic}>, subgroup_m_count = {self.subgroup_m_count}, subgroup_n_count = {self.subgroup_n_count}>"
        repl1 = f'LLVMGPUVectorDistribute workgroup_size = [{", ".join(map(str, self.workgroup_size))}] subgroup_size = {self.subgroup_size},'
        repl2 = f'tile_sizes = [[{", ".join(map(str, tile_sizes))}]]'
        repl3 = f"gpu_pipeline_options = {self.gpu_pipeline_options}"
        repl4 = f'"amdgpu-waves-per-eu" = "{self.waves_per_eu}"'

        new_mlir = ""
        for line in template:
            if "intrinsic =" in line:
                line = re.sub(expr0, repl0, line)
            if "LLVMGPUVectorDistribute " in line:
                line = re.sub(expr1, repl1, line)
            if "tile_sizes" in line:
                line = re.sub(expr2, repl2, line)
            if "gpu_pipeline_options =" in line:
                line = re.sub(expr3, repl3, line)
            if "amdgpu-waves-per-eu" in line:
                line = re.sub(expr4, repl4, line)
            new_mlir += line

        return new_mlir


@dataclass
class LLVMCPUConfiguration(BaseConfiguration):
    def get_mlir_config(self, tile_sizes: list[int]) -> str:
        base_config = self.get_base_mlir_config(tile_sizes)

        backend_config = f"""
        translation_info = #iree_codegen.translation_info<Mmt4dTilingExpert>
        > -> !transform.any_param
    """

        return base_config + backend_config

    def apply_configuration(self, template: list[str], tile_sizes: list[int]) -> str:
        self.config_logger.info(f"Applying: {self}")

        expr0 = re.compile(r"tile_sizes = \[\[([0-9]+)(, ([0-9]+))+\]")

        repl0 = f'tile_sizes = [[{", ".join(map(str, tile_sizes))}]'

        new_mlir = ""
        for line in template:
            if "linalg.mmt4d" in line and "tile_sizes" in line:
                line = re.sub(expr0, repl0, line)
            new_mlir += line

        return new_mlir


class MlirRegex(Enum):
    ssa_value = r"%[a-zA-Z0-9-_]+"
    tensor_type = r"tensor<(([0-9]+x)+((f|i)[0-9]+))>"
    device_target = r'#hal\.device\.target<"(?P<target>[a-zA-Z0-9-_]+)"'

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


@dataclass
class MLIRTransformation:
    """Transformation of MLIR context"""

    template: list[str]
    modified: str
    embeddable: str
