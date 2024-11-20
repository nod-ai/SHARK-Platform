# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen
from iree.compiler.dialects import iree_gpu

from .common import *
from .dispatch_constraints import *
from .dispatch_parser import *


def get_i64_attr(x: int, ctx: ir.Context):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64, ctx), x)


def get_i64_array_attr(array: list[int], ctx: ir.Context):
    index_attrs = [get_i64_attr(x, ctx) for x in array]
    return ir.ArrayAttr.get(index_attrs, ctx)


def build_vector_distribute_translation_info(
    configuration: Configuration,
    ctx: ir.Context
) -> iree_codegen.TranslationInfoAttr:
    extra_config = get_pipeline_config(configuration)
    pipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    pipelineAttr = iree_codegen.DispatchLoweringPassPipelineAttr.get(pipeline, ctx)
    # TODO(Max191): Use python bindings to generate the extra_config dict
    # instead of parsing the extra_config string. A bug in the iree_gpu python
    # bindings enum generation causes a redefinition of IteratorTypes, which
    # prevents the ReorderWorkgroupsStrategy enum from being exposed. Once this
    # enum is exposed, this can be done fully with python bindings.
    config_dict: ir.DictAttr = ir.Attribute.parse(f"{{{extra_config}}}", ctx)
    translation_info_attr = iree_codegen.TranslationInfoAttr.get(
        pass_pipeline=pipelineAttr,
        codegen_spec=None,
        workgroup_size=configuration.workgroup_size,
        subgroup_size=configuration.subgroup_size,
        configuration=config_dict,
        ctx=ctx,
    )
    return translation_info_attr


def build_vector_distribute_lowering_config(
    configuration: Configuration,
    reduction_tile_sizes: list[int],
    workgroup_tile_sizes: list[int],
    ctx: ir.Context,
) -> iree_gpu.LoweringConfigAttr:
    reduction_tile_sizes_attr = get_i64_array_attr(reduction_tile_sizes, ctx)
    workgroup_tile_sizes_attr = get_i64_array_attr(workgroup_tile_sizes, ctx)
    intrinsic_attr = ir.Attribute.parse(f"#iree_gpu.mma_layout<{configuration.intrinsic}>", ctx)
    lowering_config_dict = {
        "mma_kind" : intrinsic_attr,
        "reduction" : reduction_tile_sizes_attr,
        "workgroup" : workgroup_tile_sizes_attr,
        "subgroup_m_count" : get_i64_attr(configuration.subgroup_m_count, ctx),
        "subgroup_n_count" : get_i64_attr(configuration.subgroup_n_count, ctx),
    }
    lowering_config_dict_attr = ir.DictAttr.get(lowering_config_dict, ctx)
    lowering_config_attr = iree_gpu.LoweringConfigAttr.get(
        lowering_config_dict_attr,
        ctx
    )
    return lowering_config_attr


# TODO(Max191): Use python bindings to build the transform dialect spec module
# instead of using string formatting.
def build_td_spec(
    context: ir.Context,
    op: ir.Operation,
    lowering_config: iree_gpu.LoweringConfigAttr,
    translation_info: iree_codegen.TranslationInfoAttr,
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
            // Annotation Transform
            transform.named_sequence @apply_op_config(%op: !transform.any_op {{transform.readonly}},
                                                        %config: !transform.any_param {{transform.readonly}}) {{
                transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
                transform.yield
            }}

            // Custom Op Matcher
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

            // Entry Point
            transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {{transform.consumed}}) {{
                transform.foreach_match in %variant_op
                    @{func_name} -> @apply_op_config
                : (!transform.any_op) -> (!transform.any_op)
                transform.yield
            }}
        }}
        """
    return ir.Module.parse(spec_text, context)
