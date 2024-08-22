# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

import argparse
import logging
import math
import pickle
import re
import z3
from dataclasses import asdict, dataclass
from enum import Enum
from os import mkdir, path, makedirs
from typing import Callable
from textwrap import indent

import iree.compiler as ireec
from iree.compiler import ir
from iree.compiler.dialects import _linalg_ops_gen, _util_ops_gen

"""
Usage: ./candidate_gen.py 121.mlir -o "tuning/candidates" -l 1024 --lhs-dims=mk --rhs-dims=nk --tile-dims=mnk
"""

tune_logger = logging.getLogger("tune")


class DispatchKind(Enum):
    conv = 1
    mmt = 2
    contraction = 3
    batch_mmt = 4
    batch_matmul = 5
    broadcast_rhs_mmt = 6


@dataclass
class OpWalkResult:
    was_interrupted: bool = False
    dispatch_kind: DispatchKind | None = None


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
    input_type: ElementType
    m: int
    n: int
    k: int
    output_type: ElementType

    def __str__(self) -> str:
        input = str(self.input_type).upper()
        output = str(self.output_type).upper()
        return f"MFMA_{input}_{self.m}x{self.n}x{self.k}_{output}"

    @staticmethod
    def mfma_f16_16x16x16_f32():
        return MfmaIntrinsic(ElementType.f16, 16, 16, 16, ElementType.f32)

    @staticmethod
    def mfma_f16_32x32x8_f32():
        return MfmaIntrinsic(ElementType.f16, 32, 32, 8, ElementType.f32)

    @staticmethod
    def mfma_i8_16x16x32_i32():
        return MfmaIntrinsic(ElementType.i8, 16, 16, 32, ElementType.i32)

    @staticmethod
    def mfma_i8_32x32x16_i32():
        return MfmaIntrinsic(ElementType.i8, 32, 32, 16, ElementType.i32)

    @staticmethod
    def all():
        return [
            MfmaIntrinsic.mfma_f16_16x16x16_f32(),
            MfmaIntrinsic.mfma_f16_32x32x8_f32(),
            MfmaIntrinsic.mfma_i8_16x16x32_i32(),
            MfmaIntrinsic.mfma_i8_32x32x16_i32(),
        ]


@dataclass
class Configuration:
    subgroup_size: int
    workgroup_size: list[int]
    intrinsic: MfmaIntrinsic
    tile_sizes: list[int]
    subgroup_m_count: int
    subgroup_n_count: int
    waves_per_eu: int


class MlirRegex(str, Enum):
    ssa_value = r"%[a-zA-Z0-9-_]+"
    tensor_type = r"tensor<(([0-9]+x)+((f|i)[0-9]+))>"

    @staticmethod
    def dps_ins_two_args() -> str:
        return rf"ins\({MlirRegex.ssa_value}, {MlirRegex.ssa_value} : (?P<LHS>{MlirRegex.tensor_type}), (?P<RHS>{MlirRegex.tensor_type})\)"

    @staticmethod
    def dps_outs_one_arg() -> str:
        return rf"outs\({MlirRegex.ssa_value} : (?P<RES>{MlirRegex.tensor_type})\)"


def read_input_mlir(filename: str) -> list[str]:
    with open(filename, "r") as f:
        return f.readlines()


def get_mmt_tile_sizes(configuration: Configuration):
    return configuration.tile_sizes


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


def get_conv_tile_sizes(configuration: Configuration) -> list[int]:
    m, n, k = configuration.tile_sizes
    batch = 1
    fh = 1
    fw = 1

    oh = 1

    oc = n
    ow = m
    ic = k
    return [batch, oh, ow, oc, fh, fw, ic]


def get_contract_tile_sizes(configuration: Configuration, tile_dims: str) -> list[int]:
    m, n, k = configuration.tile_sizes
    tile_size = [1] * len(tile_dims)
    for idx, dim in enumerate(tile_dims):
        if dim == "m":
            tile_size[idx] = m
        if dim == "n":
            tile_size[idx] = n
        if dim == "k":
            tile_size[idx] = k
    return tile_size


def get_batch_mmt_tile_sizes(configuration: Configuration) -> list[int]:
    return [1] + configuration.tile_sizes


def get_pipeline_config(configuration: Configuration) -> str:
    extra_config = ", prefetch_shared_memory"
    if configuration.waves_per_eu != 2:
        extra_config += f', llvm_func_attrs = {{"amdgpu-waves-per-eu" = "{configuration.waves_per_eu}"}}'
    return extra_config


def get_transform_function_mmt(
    problem_size: ProblemSize, functionName: str, configuration: Configuration
) -> str:
    tile_sizes = ", ".join(map(str, get_mmt_tile_sizes(configuration)))

    wg_x, wg_y, wg_z = configuration.workgroup_size
    extra_config = get_pipeline_config(configuration)

    return f"""
transform.named_sequence @{functionName}(%matmul: !transform.any_op {{transform.readonly}}) -> (!transform.any_op, !transform.any_param) {{
  %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.cast_compatible_type %lhs = tensor<{problem_size.lhs_type}> : !transform.any_value
  transform.iree.match.cast_compatible_type %rhs = tensor<{problem_size.rhs_type}> : !transform.any_value
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[{tile_sizes}]]>,
    translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
      workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {configuration.subgroup_size},
      {{mma_schedule = #iree_gpu.mma_schedule<
         intrinsic = #iree_gpu.mma_layout<{configuration.intrinsic}>,
         subgroup_m_count = {configuration.subgroup_m_count}, subgroup_n_count = {configuration.subgroup_n_count}>
       {extra_config}}}>
    > -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}}
"""


# int64_t n = outputShape[0];
# int64_t oh = outputShape[1];
# int64_t ow = outputShape[2];
# int64_t oc = outputShape[3];
# int64_t fh = filterShape[0];
# int64_t fw = filterShape[1];
# int64_t ic = filterShape[2];
def get_transform_function_conv(
    problem_size: ProblemSize, functionName: str, configuration: Configuration
) -> str:
    dynamic_batch_input_ty = problem_size.lhs_type
    dynamic_batch_input_ty.shape = dynamic_batch_input_ty.shape.copy()
    dynamic_batch_input_ty.shape[0] = -1

    dynamic_batch_output_ty = problem_size.res_type
    dynamic_batch_output_ty.shape = dynamic_batch_output_ty.shape.copy()
    dynamic_batch_output_ty.shape[0] - 1

    input = f"tensor<{dynamic_batch_input_ty}>"
    filter = f"tensor<{problem_size.rhs_type}>"
    output = f"tensor<{dynamic_batch_output_ty}>"

    tile_sizes = ", ".join(map(str, get_conv_tile_sizes(configuration)))

    wg_x, wg_y, wg_z = configuration.workgroup_size
    extra_config = get_pipeline_config(configuration)

    return f"""
transform.named_sequence @{functionName}(%conv: !transform.any_op {{transform.readonly}})
  -> (!transform.any_op, !transform.any_param) {{
  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {{
  ^bb0(%lhs: {input}, %rhs: {filter}, %out: {output}):
    %13 = linalg.conv_2d_nhwc_hwcf {{dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}}
      ins(%lhs, %rhs : {input}, {filter})
      outs(%out : {output}) -> {output}
  }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[{tile_sizes}]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
       workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {configuration.subgroup_size},
        {{mma_schedule = #iree_gpu.mma_schedule<
            intrinsic = #iree_gpu.mma_layout<{configuration.intrinsic}>,
            subgroup_m_count = {configuration.subgroup_m_count}, subgroup_n_count = {configuration.subgroup_n_count}>
        {extra_config}}}>
    > -> !transform.any_param
  transform.yield %conv, %config : !transform.any_op, !transform.any_param
}}
"""


def get_transform_function_batch_matmul(
    problem_size: ProblemSize,
    tile_dims: str,
    functionName: str,
    configuration: Configuration,
) -> str:
    input0 = f"tensor<{problem_size.lhs_type}>"
    input1 = f"tensor<{problem_size.rhs_type}>"
    output = f"tensor<{problem_size.res_type}>"

    tile_sizes = ", ".join(map(str, get_contract_tile_sizes(configuration, tile_dims)))

    wg_x, wg_y, wg_z = configuration.workgroup_size
    extra_config = get_pipeline_config(configuration)

    return f"""
transform.named_sequence @{functionName}(%batch_matmul: !transform.any_op {{transform.readonly}})
  -> (!transform.any_op, !transform.any_param) {{
  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %batch_matmul {{
  ^bb0(%lhs: {input0}, %rhs: {input1}, %out: {output}):
    %13 = linalg.batch_matmul
      ins(%lhs, %rhs : {input0}, {input1})
      outs(%out : {output}) -> {output}
  }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[{tile_sizes}]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
       workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {configuration.subgroup_size},
        {{mma_schedule = #iree_gpu.mma_schedule<
            intrinsic = #iree_gpu.mma_layout<{configuration.intrinsic}>,
            subgroup_m_count = {configuration.subgroup_m_count}, subgroup_n_count = {configuration.subgroup_n_count}>
        {extra_config}}}>
    > -> !transform.any_param
  transform.yield %batch_matmul, %config : !transform.any_op, !transform.any_param
}}
"""


def get_transform_function_batch_mmt(
    problem_size: ProblemSize,
    functionName: str,
    configuration: Configuration,
) -> str:
    tile_sizes = ", ".join(map(str, get_batch_mmt_tile_sizes(configuration)))

    wg_x, wg_y, wg_z = configuration.workgroup_size
    extra_config = get_pipeline_config(configuration)

    return f"""
transform.named_sequence @{functionName}(%generic: !transform.any_op {{transform.readonly}}) -> (!transform.any_op, !transform.any_param) {{
  %mmt = transform.include @match_batch_mmt_i8_i8_i32 failures(propagate) (%generic) : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %generic[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %generic[1] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.cast_compatible_type %lhs = tensor<{problem_size.lhs_type}> : !transform.any_value
  transform.iree.match.cast_compatible_type %rhs = tensor<{problem_size.rhs_type}> : !transform.any_value
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[{tile_sizes}]]>,
    translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
      workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {configuration.subgroup_size},
      {{mma_schedule = #iree_gpu.mma_schedule<
         intrinsic = #iree_gpu.mma_layout<{configuration.intrinsic}>,
         subgroup_m_count = {configuration.subgroup_m_count}, subgroup_n_count = {configuration.subgroup_n_count}>
       {extra_config}}}>
    > -> !transform.any_param
  transform.yield %generic, %config : !transform.any_op, !transform.any_param
}}
"""


def get_transform_function_broadcast_rhs_mmt(
    problem_size: ProblemSize,
    functionName: str,
    configuration: Configuration,
) -> str:
    tile_sizes = ", ".join(map(str, get_batch_mmt_tile_sizes(configuration)))

    wg_x, wg_y, wg_z = configuration.workgroup_size
    extra_config = get_pipeline_config(configuration)

    lhs_dynamic_batch = problem_size.lhs_type
    lhs_dynamic_batch.shape = lhs_dynamic_batch.shape.copy()
    lhs_dynamic_batch.shape[0] = -1

    return f"""
transform.named_sequence @{functionName}(%generic: !transform.any_op {{transform.readonly}}) -> (!transform.any_op, !transform.any_param) {{
  %mmt = transform.include @match_broadcast_rhs_mmt_i8_i8_i32 failures(propagate) (%generic) : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %generic[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %generic[1] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.cast_compatible_type %lhs = tensor<{lhs_dynamic_batch}> : !transform.any_value
  transform.iree.match.cast_compatible_type %rhs = tensor<{problem_size.rhs_type}> : !transform.any_value
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[{tile_sizes}]]>,
    translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
      workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {configuration.subgroup_size},
      {{mma_schedule = #iree_gpu.mma_schedule<
         intrinsic = #iree_gpu.mma_layout<{configuration.intrinsic}>,
         subgroup_m_count = {configuration.subgroup_m_count}, subgroup_n_count = {configuration.subgroup_n_count}>
       {extra_config}}}>
    > -> !transform.any_param
  transform.yield %generic, %config : !transform.any_op, !transform.any_param
}}
"""


def apply_configuration(
    template: list[str], configuration: Configuration, tile_sizes: list[int]
) -> str:
    tune_logger.info(f"Applying: {configuration}")
    expr0 = re.compile(
        r"<intrinsic = #iree_gpu.mma_layout<(.+)>, subgroup_m_count = ([0-9]+), subgroup_n_count = ([0-9]+)>"
    )
    expr1 = re.compile(
        r"LLVMGPUVectorDistribute workgroup_size = \[.+\] subgroup_size = ([0-9]+),"
    )
    expr2 = re.compile(r"tile_sizes = \[\[([0-9]+)(, ([0-9]+))+\]\]")
    expr3 = re.compile(r"\"amdgpu-waves-per-eu\" = \"([0-9])\"")
    repl0 = f"<intrinsic = #iree_gpu.mma_layout<{configuration.intrinsic}>, subgroup_m_count = {configuration.subgroup_m_count}, subgroup_n_count = {configuration.subgroup_n_count}>"
    repl1 = f'LLVMGPUVectorDistribute workgroup_size = [{", ".join(map(str, configuration.workgroup_size))}] subgroup_size = {configuration.subgroup_size},'
    repl2 = f'tile_sizes = [[{", ".join(map(str, tile_sizes))}]]'
    repl3 = f'"amdgpu-waves-per-eu" = "{configuration.waves_per_eu}"'

    new_mlir = ""
    for line in template:
        if "intrinsic =" in line:
            line = re.sub(expr0, repl0, line)
        if "LLVMGPUVectorDistribute " in line:
            line = re.sub(expr1, repl1, line)
        if "tile_sizes" in line:
            line = re.sub(expr2, repl2, line)
        if "amdgpu-waves-per-eu" in line:
            line = re.sub(expr3, repl3, line)
        new_mlir += line

    return new_mlir


def apply_params_mmt(
    problem_size: ProblemSize, template: list[str], configuration: Configuration
) -> tuple[str, str]:
    M, N, K = problem_size.MNK
    modified = indent(
        get_transform_function_mmt(
            problem_size, f"match_mmt_{M}x{N}x{K}", configuration
        ),
        "//   ",
    )
    modified += apply_configuration(
        template, configuration, get_mmt_tile_sizes(configuration)
    )
    embeddable = indent(
        get_transform_function_mmt(problem_size, f"match_op", configuration), "  "
    )
    return modified, embeddable


def apply_params_conv(
    problem_size: ProblemSize, template: list[str], configuration: Configuration
) -> tuple[str, str]:
    conv_dims = ConvDimInfo.from_problem_size(problem_size)
    modified = indent(
        get_transform_function_conv(
            problem_size,
            f"match_conv_2d_nhwc_hwcf_Bx{conv_dims.oh}x{conv_dims.ow}x{conv_dims.oc}x{conv_dims.fh}x{conv_dims.fw}x{conv_dims.ic}",
            configuration,
        ),
        "//   ",
    )
    modified += apply_configuration(
        template, configuration, get_conv_tile_sizes(configuration)
    )
    embeddable = indent(
        get_transform_function_conv(problem_size, f"match_op", configuration),
        "  ",
    )
    return modified, embeddable


def apply_params_contract(
    problem_size: ProblemSize,
    tile_dims: str,
    template: list[str],
    configuration: Configuration,
) -> tuple[str, str]:
    # TODO: Generate transform function.
    return (
        apply_configuration(
            template, configuration, get_contract_tile_sizes(configuration, tile_dims)
        ),
        "",
    )


def apply_params_batch_matmul(
    problem_size: ProblemSize,
    tile_dims: str,
    template: list[str],
    configuration: Configuration,
) -> tuple[str, str]:
    tune_logger.info(f"{configuration}")
    M, N, K = problem_size.MNK
    modified = indent(
        get_transform_function_batch_matmul(
            problem_size,
            tile_dims,
            f"match_batch_matmul_{problem_size.matmul_size.B}x{M}x{N}x{K}",
            configuration,
        ),
        "//   ",
    )
    modified += apply_configuration(
        template, configuration, get_contract_tile_sizes(configuration, tile_dims)
    )

    embeddable = indent(
        get_transform_function_batch_matmul(
            problem_size, tile_dims, f"match_op", configuration
        ),
        "  ",
    )
    return modified, embeddable


def apply_params_batch_mmt(
    problem_size: ProblemSize, template: list[str], configuration: Configuration
) -> tuple[str, str]:
    M, N, K = problem_size.MNK
    B = problem_size.matmul_size.B
    modified = indent(
        get_transform_function_batch_mmt(
            problem_size, f"match_batch_mmt_{B}x{M}x{N}x{K}", configuration
        ),
        "//   ",
    )
    modified += apply_configuration(
        template, configuration, get_batch_mmt_tile_sizes(configuration)
    )

    embeddable = indent(
        get_transform_function_batch_mmt(problem_size, f"match_op", configuration),
        "  ",
    )
    return modified, embeddable


def apply_params_broadcast_rhs_mmt(
    problem_size: ProblemSize, template: list[str], configuration: Configuration
) -> tuple[str, str]:
    M, N, K = problem_size.MNK
    modified = indent(
        get_transform_function_broadcast_rhs_mmt(
            problem_size, f"match_broadcast_rhs_mmt_Bx{M}x{N}x{K}", configuration
        ),
        "//   ",
    )
    modified += apply_configuration(
        template, configuration, get_batch_mmt_tile_sizes(configuration)
    )

    embeddable = indent(
        get_transform_function_broadcast_rhs_mmt(
            problem_size, f"match_op", configuration
        ),
        "  ",
    )
    return modified, embeddable


def parse_tensor_type(tensor_type: str) -> ShapedType:
    shape_match = re.search(MlirRegex.tensor_type, tensor_type)
    assert shape_match

    shape_str = shape_match.group(1)
    dims_and_elem = shape_str.split("x")
    dims = [int(x) for x in dims_and_elem[:-1]]
    elem = dims_and_elem[-1]
    str_to_elem_ty = {x.name: x for x in ElementType}
    return ShapedType(dims, str_to_elem_ty[elem])


def get_shapes_mmt(template: list[str]) -> ProblemSize:
    for line in template:
        if "linalg.generic" not in line:
            continue
        if r'iterator_types = ["parallel", "parallel", "reduction"]' not in line:
            continue
        # ins(%13, %14 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%19 : tensor<2048x1280xf32>)
        mmt_re = rf"{MlirRegex.dps_ins_two_args()}\s+{MlirRegex.dps_outs_one_arg()}"
        dps = re.search(mmt_re, line)
        if dps is None:
            continue

        lhs_tensor_type = dps.group("LHS")
        rhs_tensor_type = dps.group("RHS")
        lhs_shaped_type = parse_tensor_type(lhs_tensor_type)
        assert lhs_shaped_type.rank() == 2
        lhs_M, lhs_K = lhs_shaped_type.shape

        rhs_shaped_type = parse_tensor_type(rhs_tensor_type)
        assert rhs_shaped_type.rank() == 2
        rhs_N, rhs_K = rhs_shaped_type.shape

        assert lhs_shaped_type.element_type == rhs_shaped_type.element_type
        assert lhs_K == rhs_K

        res_tensor_type = dps.group("RES")
        res_shaped_type = parse_tensor_type(res_tensor_type)
        assert res_shaped_type.rank() == 2
        res_M, res_N = res_shaped_type.shape

        assert lhs_M == res_M
        assert rhs_N == res_N

        matmul_size = MatmulSize(
            lhs_shaped_type.shape[0], rhs_shaped_type.shape[0], lhs_shaped_type.shape[1]
        )
        return ProblemSize(
            matmul_size,
            lhs_type=lhs_shaped_type,
            rhs_type=rhs_shaped_type,
            res_type=res_shaped_type,
            dispatch_kind=DispatchKind.mmt,
        )

    assert False, "Shape not found"


def get_shapes_conv(template: list[str]) -> ProblemSize:
    for line in template:
        if "linalg.conv_2d_nhwc_hwcf" not in line:
            continue

        # ins(%19, %20 : tensor<2x34x34x1280xf16>, tensor<3x3x1280x1280xf16>) outs (%27 : tensor<2x32x32x1280xf32>)
        conv_re = rf"{MlirRegex.dps_ins_two_args()}\s+{MlirRegex.dps_outs_one_arg()}"
        dps = re.search(conv_re, line)
        if dps is None:
            continue

        lhs_tensor_type = dps.group("LHS")
        rhs_tensor_type = dps.group("RHS")
        lhs_shaped_type = parse_tensor_type(lhs_tensor_type)
        assert lhs_shaped_type.rank() == 4

        rhs_shaped_type = parse_tensor_type(rhs_tensor_type)
        assert rhs_shaped_type.rank() == 4

        res_tensor_type = dps.group("RES")
        res_shaped_type = parse_tensor_type(res_tensor_type)
        assert res_shaped_type.rank() == 4

        # int64_t n = outputShape[0];
        # int64_t oh = outputShape[1];
        # int64_t ow = outputShape[2];
        # int64_t oc = outputShape[3];
        # int64_t fh = filterShape[0];
        # int64_t fw = filterShape[1];
        # int64_t ic = filterShape[2];
        dim_info = ConvDimInfo.from_rhs_res(rhs_shaped_type, res_shaped_type)
        return ProblemSize(
            MatmulSize(
                M=dim_info.oh * dim_info.ow,
                N=dim_info.oc,
                K=dim_info.fh * dim_info.fw * dim_info.ic,
                B=dim_info.n,
            ),
            lhs_shaped_type,
            rhs_shaped_type,
            res_shaped_type,
            DispatchKind.conv,
        )

    assert False, "Shape not found"


def get_shapes_contract(
    template: list[str], lhs_dims: str, rhs_dims: str
) -> ProblemSize:
    for line in template:
        if "linalg.generic" not in line:
            continue
        if "lowering_config =" not in line:
            continue
        if '"reduction"' not in line:
            continue

        # ins(%7, %8 : tensor<2x1024x1280xf16>, tensor<20x64x1280xf16>)
        cont_re = rf"{MlirRegex.dps_ins_two_args()}\s+{MlirRegex.dps_outs_one_arg()}"
        dps = re.search(cont_re, line)
        if dps is None:
            continue

        lhs_tensor_type = dps.group("LHS")
        rhs_tensor_type = dps.group("RHS")
        lhs_shaped_type = parse_tensor_type(lhs_tensor_type)
        assert lhs_shaped_type.rank() == len(lhs_dims)

        rhs_shaped_type = parse_tensor_type(rhs_tensor_type)
        assert rhs_shaped_type.rank() == len(rhs_dims)

        res_tensor_type = dps.group("RES")
        res_shaped_type = parse_tensor_type(res_tensor_type)
        assert res_shaped_type.rank() >= 2

        M = math.prod(
            val if dim == "m" else 1
            for dim, val in zip(lhs_dims, lhs_shaped_type.shape)
        )
        N = math.prod(
            val if dim == "n" else 1
            for dim, val in zip(rhs_dims, rhs_shaped_type.shape)
        )
        K0 = math.prod(
            val if dim == "k" else 1
            for dim, val in zip(lhs_dims, lhs_shaped_type.shape)
        )
        K1 = math.prod(
            val if dim == "k" else 1
            for dim, val in zip(rhs_dims, rhs_shaped_type.shape)
        )
        assert K0 == K1

        return ProblemSize(
            MatmulSize(M, N, K0),
            lhs_type=lhs_shaped_type,
            rhs_type=rhs_shaped_type,
            res_type=res_shaped_type,
            dispatch_kind=DispatchKind.contraction,
        )

    assert False, "Shape not found"


def get_shapes_batch_matmul(
    template: list[str], lhs_dims: str, rhs_dims: str
) -> ProblemSize:
    for line in template:
        if "linalg.batch_matmul" not in line:
            continue
        # ins(%9, %10 : tensor<64x72x1280xf16>, tensor<64x1280x1280xf16>)
        # outs(%12 : tensor<64x72x1280xf32>)
        cont_re = rf"{MlirRegex.dps_ins_two_args()}\s+{MlirRegex.dps_outs_one_arg()}"
        dps = re.search(cont_re, line)
        if dps is None:
            continue

        lhs_tensor_type = dps.group("LHS")
        rhs_tensor_type = dps.group("RHS")
        lhs_shaped_type = parse_tensor_type(lhs_tensor_type)
        assert lhs_shaped_type.rank() == len(lhs_dims)

        rhs_shaped_type = parse_tensor_type(rhs_tensor_type)
        assert rhs_shaped_type.rank() == len(rhs_dims)

        res_tensor_type = dps.group("RES")
        res_shaped_type = parse_tensor_type(res_tensor_type)
        assert res_shaped_type.rank() == lhs_shaped_type.rank()

        LHS = lhs_shaped_type.shape
        RHS = rhs_shaped_type.shape
        RES = res_shaped_type.shape

        B = math.prod(val if dim == "b" else 1 for dim, val in zip(lhs_dims, LHS))
        B0 = math.prod(val if dim == "b" else 1 for dim, val in zip(lhs_dims, RHS))
        B1 = math.prod(val if dim == "b" else 1 for dim, val in zip(lhs_dims, RES))
        M = math.prod(val if dim == "m" else 1 for dim, val in zip(lhs_dims, LHS))
        N = math.prod(val if dim == "n" else 1 for dim, val in zip(rhs_dims, RHS))
        K0 = math.prod(val if dim == "k" else 1 for dim, val in zip(lhs_dims, LHS))
        K1 = math.prod(val if dim == "k" else 1 for dim, val in zip(rhs_dims, RHS))
        assert B == B0 and B == B1
        assert K0 == K1

        return ProblemSize(
            MatmulSize(M, N, K0, B),
            lhs_type=lhs_shaped_type,
            rhs_type=rhs_shaped_type,
            res_type=res_shaped_type,
            dispatch_kind=DispatchKind.batch_matmul,
        )

    assert False, "Shape not found"


def get_shapes_batch_mmt(template: list[str]) -> ProblemSize:
    for line in template:
        if "linalg.generic" not in line:
            continue
        if (
            r'iterator_types = ["parallel", "parallel", "parallel", "reduction"]'
            not in line
        ):
            continue
        # ins(%11, %12 : tensor<2x4096x640xi8>, tensor<2x640x640xi8>) outs(%19 : tensor<2x4096x640xi32>)
        bmmt_re = rf"{MlirRegex.dps_ins_two_args()}\s+{MlirRegex.dps_outs_one_arg()}"
        dps = re.search(bmmt_re, line)
        if dps is None:
            continue

        lhs_tensor_type = dps.group("LHS")
        rhs_tensor_type = dps.group("RHS")
        lhs_shaped_type = parse_tensor_type(lhs_tensor_type)
        assert lhs_shaped_type.rank() == 3

        rhs_shaped_type = parse_tensor_type(rhs_tensor_type)
        assert rhs_shaped_type.rank() == 3

        res_tensor_type = dps.group("RES")
        res_shaped_type = parse_tensor_type(res_tensor_type)
        assert res_shaped_type.rank() == 3

        B0, M0, K0 = lhs_shaped_type.shape
        B1, N1, K1 = rhs_shaped_type.shape
        B2, M2, N2 = res_shaped_type.shape
        assert B0 == B1
        assert B0 == B2
        assert M0 == M2
        assert N1 == N2
        assert K0 == K1
        return ProblemSize(
            MatmulSize(M0, N1, K0, B0),
            lhs_shaped_type,
            rhs_shaped_type,
            res_shaped_type,
            DispatchKind.batch_mmt,
        )

    assert False, "Shape not found"


def is_broadcast_rhs_mmt_op(line: str) -> bool:
    if "linalg.generic" not in line:
        return False
    if (
        r'iterator_types = ["parallel", "parallel", "parallel", "reduction"]'
        not in line
    ):
        return False
    if (
        r"indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>"
        not in line
    ):
        return False
    return True


def is_broadcast_rhs_mmt(template: list[str]) -> bool:
    return any(is_broadcast_rhs_mmt_op(line) for line in template)


def get_shapes_broadcast_rhs_mmt(template: list[str]) -> ProblemSize:
    for line in template:
        if not is_broadcast_rhs_mmt_op(line):
            continue

        # ins(%11, %12 : tensor<2x1024x1280xi8>, tensor<10240x1280xi8>) outs(%19 : tensor<2x1024x10240xi32>)
        bmmt_re = rf"{MlirRegex.dps_ins_two_args()}\s+{MlirRegex.dps_outs_one_arg()}"
        dps = re.search(bmmt_re, line)
        if dps is None:
            continue

        lhs_tensor_type = dps.group("LHS")
        rhs_tensor_type = dps.group("RHS")
        lhs_shaped_type = parse_tensor_type(lhs_tensor_type)
        assert lhs_shaped_type.rank() == 3

        rhs_shaped_type = parse_tensor_type(rhs_tensor_type)
        assert rhs_shaped_type.rank() == 2

        res_tensor_type = dps.group("RES")
        res_shaped_type = parse_tensor_type(res_tensor_type)
        assert res_shaped_type.rank() == 3

        B0, M0, K0 = lhs_shaped_type.shape
        N1, K1 = rhs_shaped_type.shape
        B2, M2, N2 = res_shaped_type.shape
        assert B0 == B2
        assert M0 == M2
        assert N1 == N2
        assert K0 == K1
        return ProblemSize(
            MatmulSize(M0, N1, K0, B0),
            lhs_shaped_type,
            rhs_shaped_type,
            res_shaped_type,
            DispatchKind.broadcast_rhs_mmt,
        )

    assert False, "Shape not found"


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


def get_mfma_intrinsic_constraints(
    problem_size: ProblemSize,
    intrinsic_m: z3.ArithRef,
    intrinsic_n: z3.ArithRef,
    intrinsic_k: z3.ArithRef,
) -> z3.BoolRef:
    compatible_intrinsics = get_compatible_mfma_intrinsics(problem_size)
    assert len(compatible_intrinsics) > 0, "No compatible intrinsics found"
    return z3.Or(
        *(
            z3.And(intrinsic_m == mfma.m, intrinsic_n == mfma.n, intrinsic_k == mfma.k)
            for mfma in compatible_intrinsics
        )
    )


def get_dispatch_constraints(
    problem_size: ProblemSize,
    tile_m: z3.ArithRef,
    tile_n: z3.ArithRef,
    tile_k: z3.ArithRef,
) -> list[z3.BoolRef]:
    if problem_size.dispatch_kind != DispatchKind.conv:
        return []

    dim_info = ConvDimInfo.from_problem_size(problem_size)
    conv_constraints = []
    # WARNING: This sometimes makes the constraints UNSAT for some reason.
    conv_constraints += [tile_m <= dim_info.ow]
    conv_constraints += [tile_n <= dim_info.oc]
    conv_constraints += [tile_k <= dim_info.ic]
    return conv_constraints


def calculate_shared_memory_usage_in_bytes(
    problem_size: ProblemSize,
    m: int | z3.ArithRef,
    n: int | z3.ArithRef,
    k: int | z3.ArithRef,
) -> int | z3.ArithRef:
    lhs_memory = m * k * (problem_size.lhs_type.bitwidth // 8)
    rhs_memory = k * n * (problem_size.rhs_type.bitwidth // 8)
    return lhs_memory + rhs_memory


def generate_constraints(
    problem_size: ProblemSize,
    tile_sizes,
    num_subgroups,
    subgroup_size,
    intrinsic_size,
    workgroup_size,
    subgroup_m_count,
    subgroup_n_count,
    waves_per_eu,
):
    M, N, K = (
        problem_size.matmul_size.M,
        problem_size.matmul_size.N,
        problem_size.matmul_size.K,
    )
    m, n, k = tile_sizes
    intrinsic_mn, intrinsic_k = intrinsic_size
    wg_x, wg_y, wg_z = workgroup_size
    wg_threads = z3.Int("wg_threads")
    constraints = [wg_threads == wg_x * wg_y * wg_z]
    constraints += [subgroup_size == 64, wg_threads <= 1024]
    constraints += [
        get_mfma_intrinsic_constraints(
            problem_size, intrinsic_mn, intrinsic_mn, intrinsic_k
        )
    ]
    subgroup_k_count = 1
    constraints += [
        m >= intrinsic_mn,
        m <= 512,
        m <= M,
    ]
    constraints += [n >= intrinsic_mn, n <= 512, n <= N, N % n == 0]
    constraints += [k >= intrinsic_k, k <= 512, k <= K, K % k == 0]
    for x in (subgroup_m_count, subgroup_n_count):
        constraints += [x >= 1, x <= 32]

    subgroup_m_tile_count = z3.Int("sg_m_tcnt")
    subgroup_n_tile_count = z3.Int("sg_n_tcnt")
    subgroup_k_tile_count = z3.Int("sg_k_tcnt")
    for x in (subgroup_m_tile_count, subgroup_n_tile_count, subgroup_k_tile_count):
        constraints += [x >= 1, x <= 32]

    constraints += [m == subgroup_m_count * subgroup_m_tile_count * intrinsic_mn]
    constraints += [n == subgroup_n_count * subgroup_n_tile_count * intrinsic_mn]
    constraints += [k == subgroup_k_count * subgroup_k_tile_count * intrinsic_k]
    constraints += [wg_x == subgroup_size * subgroup_n_count]
    constraints += [wg_y == subgroup_m_count]
    constraints += [wg_z == subgroup_k_count]
    constraints += [z3.Or(wg_x <= n, wg_x <= m)]
    constraints += [k % intrinsic_mn == 0]
    constraints += [(k * n) % wg_threads == 0]
    constraints += [(k * m) % wg_threads == 0]
    subgroups = subgroup_m_count * subgroup_n_count
    if num_subgroups > 0:
        constraints += [subgroups == num_subgroups]
    else:
        constraints += [subgroups >= 1, subgroups <= 10]

    constraints += [waves_per_eu == 2]
    # constraints += [z3.Or(waves_per_eu == 2, waves_per_eu == 3, waves_per_eu == 4)]

    shared_memory = calculate_shared_memory_usage_in_bytes(problem_size, m, n, k)
    constraints += [shared_memory <= 65536]

    constraints += get_dispatch_constraints(problem_size, m, n, k)

    return constraints


def generate_solutions(problem_size: ProblemSize, num_subgrups: int):
    M, N, K = problem_size.MNK
    tune_logger.info(f"{M},{N},{K}")
    m, n, k = z3.Int("m"), z3.Int("n"), z3.Int("k")
    subgroup_size = z3.Int("subgroup_size")
    intrinsic_mn = z3.Int("intrinsic_mn")
    intrinsic_k = z3.Int("intrinsic_k")
    wg_x, wg_y, wg_z = z3.Int("wg_x"), z3.Int("wg_y"), z3.Int("wg_z")
    sg_m_cnt = z3.Int("sg_m_cnt")
    sg_n_cnt = z3.Int("sg_n_cnt")
    waves_per_eu = z3.Int("waves_per_eu")
    all_vars = [
        m,
        n,
        k,
        subgroup_size,
        intrinsic_mn,
        intrinsic_k,
        wg_x,
        wg_y,
        wg_z,
        sg_m_cnt,
        sg_n_cnt,
        waves_per_eu,
    ]

    solver = z3.Solver()
    constraints = generate_constraints(
        problem_size,
        [m, n, k],
        num_subgrups,
        subgroup_size,
        [intrinsic_mn, intrinsic_k],
        [wg_x, wg_y, wg_z],
        sg_m_cnt,
        sg_n_cnt,
        waves_per_eu,
    )
    solver.add(z3.simplify(z3.And(constraints)))
    tune_logger.debug(f"Initial constraints: {solver}")
    i = 0
    while solver.check() == z3.sat:
        model = solver.model()
        lookup = lambda var: model[var].as_long()

        config = Configuration(
            lookup(subgroup_size),
            [lookup(wg_x), lookup(wg_y), lookup(wg_z)],
            MfmaIntrinsic(
                problem_size.lhs_type.element_type,
                lookup(intrinsic_mn),
                lookup(intrinsic_mn),
                lookup(intrinsic_k),
                problem_size.res_type.element_type,
            ),
            [lookup(m), lookup(n), lookup(k)],
            lookup(sg_m_cnt),
            lookup(sg_n_cnt),
            lookup(waves_per_eu),
        )
        solver.add(z3.simplify(z3.Not(z3.And(list(x == model[x] for x in all_vars)))))
        i += 1
        yield config


def get_default_output_dir() -> str:
    from datetime import datetime

    return "tuning_" + datetime.now().strftime("%Y_%m_%d_%H_%M")


def parse_mlir(mlir_text: str) -> ir.Module:
    mlir_module = None
    with ireec.ir.Context() as context:
        try:
            mlir_module = ireec.ir.Module.parse(mlir_text)
            tune_logger.info("MLIR parsing successful!")
        except ireec.ir.MLIRError as e:
            tune_logger.error(f"Error parsing MLIR: {e}")
            raise RuntimeError(f"Error parsing MLIR: {e}")

    return mlir_module


def walk_callback_detect_type(
    op: ir.Operation, walk_result: OpWalkResult
) -> ir.WalkResult:
    if op.name == "linalg.conv_2d_nhwc_hwcf":
        walk_result.was_interrupted = True
        walk_result.dispatch_kind = DispatchKind.conv
        return ir.WalkResult.INTERRUPT

    if op.name == "util.func":
        func_name = str(op.opview.sym_name)
        if "batch_matmul_transpose_b" in func_name:
            walk_result.was_interrupted = True
            walk_result.dispatch_kind = DispatchKind.batch_mmt
            return ir.WalkResult.INTERRUPT
        if "batch_matmul" in func_name:
            walk_result.was_interrupted = True
            walk_result.dispatch_kind = DispatchKind.batch_matmul
            return ir.WalkResult.INTERRUPT
        if "matmul_transpose_b" in func_name:
            walk_result.was_interrupted = True
            walk_result.dispatch_kind = DispatchKind.mmt
            return ir.WalkResult.INTERRUPT
        if "matmul_like" in func_name:
            walk_result.was_interrupted = True
            walk_result.dispatch_kind = DispatchKind.contraction
            return ir.WalkResult.INTERRUPT
    return ir.WalkResult.ADVANCE


def walk_mlir_op(mlir_module: ir.Module) -> OpWalkResult:
    walk_result = OpWalkResult()
    for op in mlir_module.body.operations:
        op.walk(
            lambda op: walk_callback_detect_type(op, walk_result),
            ir.WalkOrder.POST_ORDER,
        )
        if walk_result.was_interrupted:
            break

    return walk_result


def tune(
    input: str,
    output: str = "",
    limit: int = 4096,
    num_subgroups: int = 4,
    lhs_dims: str = "mk",
    rhs_dims: str = "nk",
    tile_dims: str = "mnk",
):
    input_file = str(input)

    if not output:
        output = get_default_output_dir()

    # Create the directory if it does not exist
    makedirs(str(output), exist_ok=True)

    tune_logger.debug(f"Output directory {output}")
    tune_logger.debug(f"Processing {input_file}")
    mlir_template = read_input_mlir(input_file)
    mlir_text = "".join(mlir_template)

    mlir_module = parse_mlir(mlir_text)
    walk_result = walk_mlir_op(mlir_module)
    assert walk_result.dispatch_kind != None

    # Save the input file as the first candidate.
    with open(path.join(output, f"0.mlir"), "w") as f:
        f.write(mlir_text)

    get_shapes_fn: Callable[[list[str]], ProblemSize] | None = None
    apply_params_fn: (
        Callable[[ProblemSize, list[str], Configuration], tuple[str, str]] | None
    ) = None
    if walk_result.dispatch_kind == DispatchKind.conv:
        get_shapes_fn = get_shapes_conv
        apply_params_fn = apply_params_conv
    elif walk_result.dispatch_kind == DispatchKind.mmt:
        get_shapes_fn = get_shapes_mmt
        apply_params_fn = apply_params_mmt
    elif walk_result.dispatch_kind == DispatchKind.contraction:
        if is_broadcast_rhs_mmt(mlir_template):
            get_shapes_fn = get_shapes_broadcast_rhs_mmt
            apply_params_fn = apply_params_broadcast_rhs_mmt
        else:
            get_shapes_fn = lambda template: get_shapes_contract(
                template, lhs_dims, rhs_dims
            )
            apply_params_fn = lambda ps, template, config: apply_params_contract(
                ps, tile_dims, template, config
            )
    elif walk_result.dispatch_kind == DispatchKind.batch_matmul:
        get_shapes_fn = lambda template: get_shapes_batch_matmul(
            template, lhs_dims, rhs_dims
        )
        apply_params_fn = lambda ps, template, config: apply_params_batch_matmul(
            ps, tile_dims, template, config
        )
    elif walk_result.dispatch_kind == DispatchKind.batch_mmt:
        get_shapes_fn = get_shapes_batch_mmt
        apply_params_fn = apply_params_batch_mmt
    else:
        assert False, f"Unhandled dispatch kind: {walk_result.dispatch_kind}"

    problem_size = get_shapes_fn(mlir_template)
    tune_logger.debug(str(problem_size))

    configs = []
    for i, config in enumerate(generate_solutions(problem_size, num_subgroups)):
        if i >= limit:
            break
        tune_logger.info(f"Solution #{i+1}: {config}")
        configs.append(config)
        new_mlir, embeddable_tuning = apply_params_fn(
            problem_size, mlir_template, config
        )

        with open(path.join(output, f"{i+1}.mlir"), "w") as f:
            f.write(new_mlir)
        with open(path.join(output, f"{i+1}_config.mlir"), "w") as f:
            f.write(embeddable_tuning)

    with open(path.join(output, "configs.pkl"), "wb") as file:
        pickle.dump(configs, file)

    tune_logger.info(f"Generated {len(configs)} candidates")
    tune_logger.info(f"Configurations .pkl is stored in {output}/configs.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input mlir file", type=str)
    parser.add_argument(
        "-o", "--output", help="Output dir", type=str, default=get_default_output_dir()
    )
    parser.add_argument(
        "-l",
        "--limit",
        help="Max number of candidates generated",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--num-subgroups",
        help="Number of subgroups per workgroup to use. (-1 == unconstrained)",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--lhs-dims", help="Map of LHS matmul dims", type=str, default="mk"
    )
    parser.add_argument(
        "--rhs-dims", help="Map of RHS matmul dims", type=str, default="nk"
    )
    parser.add_argument(
        "--tile-dims", help="Map of tile size matmul dims", type=str, default="mnk"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output to stdout"
    )

    args = parser.parse_args()
    tune_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # Create printing formatter for logging info
    formatter = logging.Formatter("%(message)s")

    # Create a handler to print to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    tune_logger.addHandler(console_handler)

    # # Optionally, add a file handler to log to a file
    # file_handler = logging.FileHandler("tune.log")
    # file_handler.setFormatter(formatter)
    # tune_logger.addHandler(file_handler)

    tune(
        args.input,
        args.output,
        args.limit,
        args.num_subgroups,
        args.lhs_dims,
        args.rhs_dims,
        args.tile_dims,
    )


if __name__ == "__main__":
    args = main()
