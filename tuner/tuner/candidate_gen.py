# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

"""
Generate candidates by tweaking op configuration for tuning.

It can be invoked in two ways:
    1. From another python script, import and call `tune()`
    2. Run this script directly from the command

Usage: ./candidate_gen.py 121.mlir -o "tuning/candidates" -l 1024 --lhs-dims=mk --rhs-dims=nk --tile-dims=mnk

"""

import argparse
import logging
import pickle
import re
from dataclasses import dataclass
from os import path, makedirs
from typing import Optional
from textwrap import indent
from abc import abstractmethod

from iree.compiler import ir  # type: ignore

from .common import *
from .dispatch_constraints import *
from .dispatch_parser import *

tune_logger = logging.getLogger("tune")


def apply_configuration(
    template: list[str], configuration: Configuration, tile_sizes: list[int]
) -> str:
    tune_logger.info(f"Applying: {configuration}")
    expr0 = re.compile(
        r"<intrinsic = #iree_gpu\.mma_layout<(.+)>, subgroup_m_count = ([0-9]+), subgroup_n_count = ([0-9]+)>"
    )
    expr1 = re.compile(
        r"LLVMGPUVectorDistribute workgroup_size = \[.+\] subgroup_size = ([0-9]+),"
    )
    expr2 = re.compile(r"tile_sizes = \[\[([0-9]+)(, ([0-9]+))+\]\]")
    expr3 = re.compile(r"gpu_pipeline_options = #iree_gpu\.pipeline_options<([^>]*)>")
    expr4 = re.compile(r"\"amdgpu-waves-per-eu\" = \"([0-9])\"")
    repl0 = f"<intrinsic = #iree_gpu.mma_layout<{configuration.intrinsic}>, subgroup_m_count = {configuration.subgroup_m_count}, subgroup_n_count = {configuration.subgroup_n_count}>"
    repl1 = f'LLVMGPUVectorDistribute workgroup_size = [{", ".join(map(str, configuration.workgroup_size))}] subgroup_size = {configuration.subgroup_size},'
    repl2 = f'tile_sizes = [[{", ".join(map(str, tile_sizes))}]]'
    repl3 = f"gpu_pipeline_options = {configuration.gpu_pipeline_options}"
    repl4 = f'"amdgpu-waves-per-eu" = "{configuration.waves_per_eu}"'

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


def build_td_spec(
    ir_module: ir.Module,
    op: ir.Operation,
    lowering_config: str,
    translation_info: str,
    func_name: str,
) -> ir.Module:
    bbargs = []
    for operand in op.operands:
        ssa_name = operand.get_name()
        operand_type = operand.type
        bbargs.append(f"{ssa_name}: {operand_type}")
    bbargs_str = ", ".join(bbargs)
    root_operation = str(op)
    spec_text = f"""
        module attributes {{ transform.with_named_sequence }} {{
        transform.named_sequence @{func_name}(%cont: !transform.any_op {{transform.readonly}})
            -> (!transform.any_op, !transform.any_param) {{
            %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %cont {{
            ^bb0({bbargs_str}):
            {root_operation}
            }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
            %config = transform.param.constant #iree_codegen.compilation_info<
            lowering_config = {lowering_config},
            translation_info = {translation_info}
            > -> !transform.any_param
            transform.yield %cont, %config : !transform.any_op, !transform.any_param
        }}
        }}
        """
    return ir.Module.parse(spec_text, ir_module.context)


def build_vector_distribute_translation_info(configuration: Configuration) -> str:
    wg_x, wg_y, wg_z = configuration.workgroup_size
    extra_config = get_pipeline_config(configuration)
    translation_info = f"""
        #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
            workgroup_size = [{wg_x}, {wg_y}, {wg_z}]
            subgroup_size = {configuration.subgroup_size},
            {{{extra_config}}}>"""
    return translation_info


class DispatchTuner(DispatchParser):
    # TODO(https://github.com/nod-ai/shark-ai/issues/453): Remove this in favor of configuring using transform dialect.
    @abstractmethod
    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: Configuration,
    ) -> MLIRTransformation:
        """Apply parameter transformations to the operation."""
        pass
    @abstractmethod
    def get_td_spec(
        self,
        ir_module: ir.Module,
        configuration: Configuration,
    ) -> ir.Module:
        """Generate a transform dialect spec module for the funcOp."""
        pass


class DispatchTunerRegistry:
    def __init__(self):
        self.registry = set()

    def register(self, dispatch_tuners: list[DispatchTuner]) -> None:
        for dispatch_tuner in dispatch_tuners:
            self.registry.add(dispatch_tuner)

    def validate_translation(self, attrs: list[ir.NamedAttribute]) -> bool:
        for attr in attrs:
            if (attr.name == "translation_info") and (
                "LLVMGPUVectorDistribute" in str(attr.attr)
            ):
                return True
        assert False, "Translation info not supported"

    def find_handler(self, op_name: str) -> DispatchTuner:
        for dispatch_tuner in self.registry:
            if dispatch_tuner.supports(op_name):
                return dispatch_tuner
        assert False, "Dispatch kind not supported"


class MmtTuner(DispatchTuner, MmtParser):
    def get_transform_function_mmt(
        self, problem_size: ProblemSize, functionName: str, configuration: Configuration
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

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: Configuration,
    ) -> MLIRTransformation:
        M, N, K = problem_size.MNK
        modified = indent(
            self.get_transform_function_mmt(
                problem_size, f"match_mmt_{M}x{N}x{K}", configuration
            ),
            "//   ",
        )
        modified += apply_configuration(
            template, configuration, get_mmt_tile_sizes(configuration)
        )
        embeddable = indent(
            self.get_transform_function_mmt(problem_size, f"match_op", configuration),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)

    def get_reduction_tile_sizes(self, configuration: Configuration):
        tile_sizes = [0, 0]
        tile_sizes.append(configuration.tile_sizes[2])
        return tile_sizes

    def get_workgroup_tile_sizes(self, configuration: Configuration):
        tile_sizes = configuration.tile_sizes[:2]
        tile_sizes.append(0)
        return tile_sizes

    def build_vector_distribute_lowering_config(
        self,
        configuration: Configuration,
    ) -> str:
        reduction_tile_sizes = ", ".join(map(str, self.get_reduction_tile_sizes(configuration)))
        workgroup_tile_sizes = ", ".join(map(str, self.get_workgroup_tile_sizes(configuration)))
        lowering_config = f"""
            #iree_gpu.lowering_config<{{
                mma_kind = #iree_gpu.mma_layout<{configuration.intrinsic}>,
                subgroup_m_count = {configuration.subgroup_m_count},
                subgroup_n_count = {configuration.subgroup_n_count},
                reduction = [{reduction_tile_sizes}],
                workgroup = [{workgroup_tile_sizes}]}}>"""
        return lowering_config

    def get_td_spec(
        self,
        ir_module: ir.Module,
        configuration: Configuration,
    ) -> ir.Module:
        mmt_op: ir.Operation = self.get_mmt_operation(ir_module)
        lowering_config = self.build_vector_distribute_lowering_config(configuration)
        translation_info = build_vector_distribute_translation_info(configuration)
        lhs_type = ir.ShapedType(mmt_op.operands[0].type)
        rhs_type = ir.ShapedType(mmt_op.operands[1].type)
        acc_type = ir.ShapedType(mmt_op.operands[2].type)
        M = acc_type.get_dim_size(0)
        N = acc_type.get_dim_size(1)
        K = lhs_type.get_dim_size(1)
        func_name = f"match_mmt_{M}x{N}x{K}_{lhs_type.element_type}x{rhs_type.element_type}x{acc_type.element_type}"
        return build_td_spec(ir_module, mmt_op, lowering_config, translation_info, func_name)


class ConvTuner(DispatchTuner, ConvParser):
    # int64_t n = outputShape[0];
    # int64_t oh = outputShape[1];
    # int64_t ow = outputShape[2];
    # int64_t oc = outputShape[3];
    # int64_t fh = filterShape[0];
    # int64_t fw = filterShape[1];
    # int64_t ic = filterShape[2];
    def get_transform_function_conv(
        self, problem_size: ProblemSize, functionName: str, configuration: Configuration
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

        tile_sizes = ", ".join(map(str, self.get_conv_tile_sizes(configuration)))

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

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: Configuration,
    ) -> MLIRTransformation:
        conv_dims = ConvDimInfo.from_problem_size(problem_size)
        modified = indent(
            self.get_transform_function_conv(
                problem_size,
                f"match_conv_2d_nhwc_hwcf_Bx{conv_dims.oh}x{conv_dims.ow}x{conv_dims.oc}x{conv_dims.fh}x{conv_dims.fw}x{conv_dims.ic}",
                configuration,
            ),
            "//   ",
        )
        modified += apply_configuration(
            template, configuration, self.get_conv_tile_sizes(configuration)
        )
        embeddable = indent(
            self.get_transform_function_conv(problem_size, f"match_op", configuration),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)

    def get_reduction_tile_sizes(self, configuration: Configuration):
        tile_sizes = [0, 0, 0, 0]
        tile_sizes.extend(configuration.tile_sizes[4:])
        return tile_sizes

    def get_workgroup_tile_sizes(self, configuration: Configuration):
        tile_sizes = configuration.tile_sizes[:4]
        tile_sizes.extend([0, 0, 0])
        return tile_sizes

    def build_vector_distribute_lowering_config(
        self,
        configuration: Configuration,
    ) -> str:
        reduction_tile_sizes = ", ".join(map(str, self.get_reduction_tile_sizes(configuration)))
        workgroup_tile_sizes = ", ".join(map(str, self.get_workgroup_tile_sizes(configuration)))
        lowering_config = f"""
            #iree_gpu.lowering_config<{{
                mma_kind = #iree_gpu.mma_layout<{configuration.intrinsic}>,
                subgroup_m_count = {configuration.subgroup_m_count},
                subgroup_n_count = {configuration.subgroup_n_count},
                reduction = [{reduction_tile_sizes}],
                workgroup = [{workgroup_tile_sizes}]}}>"""
        return lowering_config

    def get_td_spec(
        self,
        ir_module: ir.Module,
        configuration: Configuration,
    ) -> ir.Module:
        conv_op: ir.Operation = self.get_conv_operation(ir_module)
        lowering_config = self.build_vector_distribute_lowering_config(configuration)
        translation_info = build_vector_distribute_translation_info(configuration)
        lhs_type = ir.ShapedType(conv_op.operands[0].type)
        rhs_type = ir.ShapedType(conv_op.operands[1].type)
        acc_type = ir.ShapedType(conv_op.operands[2].type)
        N = acc_type.get_dim_size(0)
        H = acc_type.get_dim_size(1)
        W = acc_type.get_dim_size(2)
        C = rhs_type.get_dim_size(2)
        P = rhs_type.get_dim_size(0)
        Q = rhs_type.get_dim_size(1)
        F = rhs_type.get_dim_size(3)
        conv_type = conv_op.name.split(".")[-1]
        func_name = f"match_{conv_type}_{N}x{H}x{W}x{C}x{P}x{Q}x{F}_{lhs_type.element_type}x{rhs_type.element_type}x{acc_type.element_type}"
        return build_td_spec(ir_module, conv_op, lowering_config, translation_info, func_name)

class ContractionTuner(DispatchTuner, ContractionParser):
    def get_transform_function_broadcast_rhs_mmt(
        self,
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

    def apply_params_broadcast_rhs_mmt(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: Configuration,
    ) -> MLIRTransformation:
        M, N, K = problem_size.MNK
        modified = indent(
            self.get_transform_function_broadcast_rhs_mmt(
                problem_size, f"match_broadcast_rhs_mmt_Bx{M}x{N}x{K}", configuration
            ),
            "//   ",
        )
        modified += apply_configuration(
            template, configuration, get_batch_mmt_tile_sizes(configuration)
        )

        embeddable = indent(
            self.get_transform_function_broadcast_rhs_mmt(
                problem_size, f"match_op", configuration
            ),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: Configuration,
    ) -> MLIRTransformation:
        if self.is_broadcast_rhs_mmt(template):
            return self.apply_params_broadcast_rhs_mmt(
                problem_size, template, configuration
            )

        # TODO: Generate transform function.
        return MLIRTransformation(
            template,
            apply_configuration(
                template,
                configuration,
                get_contract_tile_sizes(configuration, self.tile_dims),
            ),
            "",
        )

    def get_td_spec(
        self,
        ir_module: ir.Module,
        configuration: Configuration,
    ) -> ir.Module:
        pass

class BatchMmtTuner(DispatchTuner, BatchMmtParser):
    def get_transform_function_batch_mmt(
        self,
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

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: Configuration,
    ) -> MLIRTransformation:
        M, N, K = problem_size.MNK
        B = problem_size.matmul_size.B
        modified = indent(
            self.get_transform_function_batch_mmt(
                problem_size, f"match_batch_mmt_{B}x{M}x{N}x{K}", configuration
            ),
            "//   ",
        )
        modified += apply_configuration(
            template, configuration, get_batch_mmt_tile_sizes(configuration)
        )

        embeddable = indent(
            self.get_transform_function_batch_mmt(
                problem_size, f"match_op", configuration
            ),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)

    def get_td_spec(
        self,
        ir_module: ir.Module,
        configuration: Configuration,
    ) -> ir.Module:
        pass

class BatchMatmulTuner(DispatchTuner, BatchMatmulParser):
    def get_transform_function_batch_matmul(
        self,
        problem_size: ProblemSize,
        tile_dims: str,
        functionName: str,
        configuration: Configuration,
    ) -> str:
        input0 = f"tensor<{problem_size.lhs_type}>"
        input1 = f"tensor<{problem_size.rhs_type}>"
        output = f"tensor<{problem_size.res_type}>"

        tile_sizes = ", ".join(
            map(str, get_contract_tile_sizes(configuration, tile_dims))
        )

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

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: Configuration,
    ) -> MLIRTransformation:
        M, N, K = problem_size.MNK
        modified = indent(
            self.get_transform_function_batch_matmul(
                problem_size,
                self.tile_dims,
                f"match_batch_matmul_{problem_size.matmul_size.B}x{M}x{N}x{K}",
                configuration,
            ),
            "//   ",
        )
        modified += apply_configuration(
            template,
            configuration,
            get_contract_tile_sizes(configuration, self.tile_dims),
        )

        embeddable = indent(
            self.get_transform_function_batch_matmul(
                problem_size, self.tile_dims, f"match_op", configuration
            ),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)

    def get_td_spec(
        self,
        ir_module: ir.Module,
        configuration: Configuration,
    ) -> ir.Module:
        pass

@dataclass
class OpWalkResult:
    was_interrupted: bool = False
    dispatch_tuner: Optional[DispatchTuner] = None


def walk_callback_get_fn(
    op: ir.Operation,
    walk_result: OpWalkResult,
    dispatch_tuner_registry: DispatchTunerRegistry,
) -> ir.WalkResult:
    if op.name == "func.func":
        dispatch_tuner_registry.validate_translation([a for a in op.opview.attributes])
    if op.name == "util.func":
        func_name = str(op.opview.sym_name)
        walk_result.was_interrupted = True
        walk_result.dispatch_tuner = dispatch_tuner_registry.find_handler(func_name)
        return ir.WalkResult.INTERRUPT
    return ir.WalkResult.ADVANCE


def walk_mlir_op(
    mlir_module: ir.Module,
    dispatch_tuner_registry: DispatchTunerRegistry,
) -> OpWalkResult:
    walk_result = OpWalkResult()
    for op in mlir_module.body.operations:
        op.walk(
            lambda op: walk_callback_get_fn(op, walk_result, dispatch_tuner_registry),
            ir.WalkOrder.POST_ORDER,
        )
        if walk_result.was_interrupted:
            break
    return walk_result


def get_default_output_dir() -> str:
    from datetime import datetime

    return "tuning_" + datetime.now().strftime("%Y_%m_%d_%H_%M")


# TODO(Max191): Remove in favor of using tune_with_td.
def tune(
    input: str,  # Path to the mlir file to be tuned
    output: str = "",  # Path to the output directory, auto creates one if not given
    limit: int = 4096,  # Max candidates to be generated
    num_subgroups: int = 4,  # GPU spec, used to determine candidate generation constraints
    lhs_dims: str = "mk",  # Dimensions for the left-hand side operand in matrix operations
    rhs_dims: str = "nk",  # Dimensions for the right-hand side operand in matrix operations
    tile_dims: str = "mnk",  # Dimensions for the tile size
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

    with ir.Context() as ctx:
        tuner_context = TunerContext(ctx, tune_logger)
        mlir_module: ir.Module = parse_mlir(mlir_text, tuner_context)
        # Save the input file as the first candidate.
        with open(path.join(output, f"0.mlir"), "w") as f:
            f.write(mlir_text)

        dispatch_tuner_registry = DispatchTunerRegistry()
        dispatch_tuner_registry.register(
            [
                MmtTuner(),
                ConvTuner(),
                ContractionTuner(lhs_dims, rhs_dims, tile_dims),
                BatchMmtTuner(),
                BatchMatmulTuner(lhs_dims, rhs_dims, tile_dims),
            ]
        )

        walk_result: OpWalkResult = walk_mlir_op(mlir_module, dispatch_tuner_registry)

        dispatch_tuner = walk_result.dispatch_tuner
        assert dispatch_tuner, "No suitable dispatch tuner found"
        problem_size: ProblemSize = dispatch_tuner.get_shapes(mlir_template)
        tune_logger.debug(str(problem_size))
        configs = []
        for i, config in enumerate(
            generate_solutions(tuner_context, problem_size, num_subgroups)
        ):
            if i >= limit:
                break
            tune_logger.info(f"Solution #{i+1}: {config}")
            configs.append(config)
            tf_mlir = dispatch_tuner.apply_params(problem_size, mlir_template, config)

            with open(path.join(output, f"{i+1}.mlir"), "w") as f:
                f.write(tf_mlir.modified)
            with open(path.join(output, f"{i+1}_config.mlir"), "w") as f:
                f.write(tf_mlir.embeddable)

        with open(path.join(output, "configs.pkl"), "wb") as file:
            pickle.dump(configs, file)

        tune_logger.info(f"Generated {len(configs)} candidates")
        tune_logger.info(f"Configurations .pkl is stored in {output}/configs.pkl")


def tune_with_td(
    input: str,  # Path to the mlir file to be tuned
    output: str = "",  # Path to the output directory, auto creates one if not given
    limit: int = 4096,  # Max candidates to be generated
    num_subgroups: int = 4,  # GPU spec, used to determine candidate generation constraints
    lhs_dims: str = "mk",  # Dimensions for the left-hand side operand in matrix operations
    rhs_dims: str = "nk",  # Dimensions for the right-hand side operand in matrix operations
    tile_dims: str = "mnk",  # Dimensions for the tile size
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

    with ir.Context() as ctx:
        tuner_context = TunerContext(ctx, tune_logger)
        mlir_module: ir.Module = parse_mlir(mlir_text, tuner_context)

        dispatch_tuner_registry = DispatchTunerRegistry()
        dispatch_tuner_registry.register(
            [
                MmtTuner(),
                ConvTuner(),
                # NYI: ContractionTuner, BatchMmtTuner, and BatchMatmulTuner
                # ContractionTuner(lhs_dims, rhs_dims, tile_dims),
                # BatchMmtTuner(),
                # BatchMatmulTuner(lhs_dims, rhs_dims, tile_dims),
            ]
        )

        walk_result: OpWalkResult = walk_mlir_op(mlir_module, dispatch_tuner_registry)

        dispatch_tuner = walk_result.dispatch_tuner
        assert dispatch_tuner, "No suitable dispatch tuner found"
        problem_size: ProblemSize = dispatch_tuner.get_shapes(mlir_template)
        tune_logger.debug(str(problem_size))

        # Save an empty module as the first candidate, and set its config to
        # None.
        with open(path.join(output, f"0_spec.mlir"), "w") as f:
            default_td_spec = "module {}"
            f.write(default_td_spec)

        configs = [None]
        for i, config in enumerate(
            generate_solutions(tuner_context, problem_size, num_subgroups)
        ):
            if i >= limit:
                break
            tune_logger.info(f"Solution #{i+1}: {config}")
            configs.append(config)
            tf_spec = dispatch_tuner.get_td_spec(mlir_module, config)
            if tf_spec is None:
                continue
            with open(path.join(output, f"{i+1}_spec.mlir"), "w") as f:
                f.write(str(tf_spec))

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
