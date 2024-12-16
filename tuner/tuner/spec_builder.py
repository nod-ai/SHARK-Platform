# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen  # type: ignore

from .common import *
from .dispatch_constraints import *
from .dispatch_parser import *


# TODO(Max191): Use python bindings to build the transform dialect spec module
# instead of using string formatting.
def build_td_spec(
    context: ir.Context,
    op: ir.Operation,
    compilation_info: iree_codegen.CompilationInfoAttr,
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
                %config = transform.param.constant {compilation_info} -> !transform.any_param
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
