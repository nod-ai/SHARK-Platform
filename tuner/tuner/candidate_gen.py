# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

"""z
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
from typing import Optional, Generator
from textwrap import indent
from abc import abstractmethod

from iree.compiler import ir  # type: ignore

from iree.compiler.dialects import iree_codegen  # type: ignore

from .common import *
from .dispatch_constraints import *
from .dispatch_parser import *

tune_logger = logging.getLogger("tune")


class DispatchTuner(DispatchParser):
    # TODO(https://github.com/nod-ai/shark-ai/issues/453): Remove this in favor of configuring using transform dialect.
    @abstractmethod
    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: BaseConfiguration,
    ) -> MLIRTransformation:
        """Apply parameter transformations to the operation."""
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

            if (attr.name == "translation_info") and (
                "Mmt4dTilingExpert" in str(attr.attr)
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
        self,
        problem_size: ProblemSize,
        functionName: str,
        configuration: BaseConfiguration,
    ) -> str:
        tile_sizes = get_mmt_tile_sizes(configuration)
        config_mlir = configuration.get_mlir_config(tile_sizes)

        return f"""
    transform.named_sequence @{functionName}(%matmul: !transform.any_op {{transform.readonly}}) -> (!transform.any_op, !transform.any_param) {{
    %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<{problem_size.lhs_type}> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<{problem_size.rhs_type}> : !transform.any_value
    {config_mlir}
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
    }}
    """

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: BaseConfiguration,
    ) -> MLIRTransformation:
        M, N, K = problem_size.MNK
        modified = indent(
            self.get_transform_function_mmt(
                problem_size, f"match_mmt_{M}x{N}x{K}", configuration
            ),
            "//   ",
        )
        modified += configuration.apply_configuration(
            template, get_mmt_tile_sizes(configuration)
        )
        embeddable = indent(
            self.get_transform_function_mmt(problem_size, f"match_op", configuration),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)


class Mmt4dTuner(DispatchTuner, Mmt4dParser):
    def get_transform_function_mmt4d(
        self,
        problem_size: ProblemSize,
        functionName: str,
        configuration: BaseConfiguration,
    ) -> str:
        tile_sizes = get_mmt_tile_sizes(configuration)
        config_mlir = configuration.get_mlir_config(tile_sizes)

        return f"""
    transform.named_sequence @{functionName}(%matmul: !transform.any_op {{transform.readonly}}) -> (!transform.any_op, !transform.any_param) {{
    %mmt = transform.include @match_mmt4d_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<{problem_size.lhs_type}> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<{problem_size.rhs_type}> : !transform.any_value
    {config_mlir}
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
    }}
    """

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: BaseConfiguration,
    ) -> MLIRTransformation:
        M, N, K = problem_size.MNK
        modified = indent(
            self.get_transform_function_mmt4d(
                problem_size, f"match_mmt4d_{M}x{N}x{K}", configuration
            ),
            "//   ",
        )
        modified += configuration.apply_configuration(
            template, get_mmt_tile_sizes(configuration)
        )
        embeddable = indent(
            self.get_transform_function_mmt4d(problem_size, f"match_op", configuration),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)


class ConvTuner(DispatchTuner, ConvParser):
    # int64_t n = outputShape[0];
    # int64_t oh = outputShape[1];
    # int64_t ow = outputShape[2];
    # int64_t oc = outputShape[3];
    # int64_t fh = filterShape[0];
    # int64_t fw = filterShape[1];
    # int64_t ic = filterShape[2];
    def get_transform_function_conv(
        self,
        problem_size: ProblemSize,
        functionName: str,
        configuration: BaseConfiguration,
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

        tile_sizes = self.get_conv_tile_sizes(configuration)
        config_mlir = configuration.get_mlir_config(tile_sizes)

        return f"""
    transform.named_sequence @{functionName}(%conv: !transform.any_op {{transform.readonly}})
    -> (!transform.any_op, !transform.any_param) {{
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {{
    ^bb0(%lhs: {input}, %rhs: {filter}, %out: {output}):
        %13 = linalg.conv_2d_nhwc_hwcf {{dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}}
        ins(%lhs, %rhs : {input}, {filter})
        outs(%out : {output}) -> {output}
    }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
        {config_mlir}
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
    }}
    """

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: BaseConfiguration,
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
        modified += configuration.apply_configuration(
            template, self.get_conv_tile_sizes(configuration)
        )
        embeddable = indent(
            self.get_transform_function_conv(problem_size, f"match_op", configuration),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)


class ContractionTuner(DispatchTuner, ContractionParser):
    def get_transform_function_broadcast_rhs_mmt(
        self,
        problem_size: ProblemSize,
        functionName: str,
        configuration: BaseConfiguration,
    ) -> str:
        tile_sizes = get_batch_mmt_tile_sizes(configuration)
        config_mlir = configuration.get_mlir_config(tile_sizes)

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
{config_mlir}
transform.yield %generic, %config : !transform.any_op, !transform.any_param
}}
"""

    def apply_params_broadcast_rhs_mmt(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: BaseConfiguration,
    ) -> MLIRTransformation:
        M, N, K = problem_size.MNK
        modified = indent(
            self.get_transform_function_broadcast_rhs_mmt(
                problem_size, f"match_broadcast_rhs_mmt_Bx{M}x{N}x{K}", configuration
            ),
            "//   ",
        )
        modified += configuration.apply_configuration(
            template, get_batch_mmt_tile_sizes(configuration)
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
        configuration: BaseConfiguration,
    ) -> MLIRTransformation:
        if self.is_broadcast_rhs_mmt(template):
            return self.apply_params_broadcast_rhs_mmt(
                problem_size, template, configuration
            )

        # TODO: Generate transform function.
        return MLIRTransformation(
            template,
            configuration.apply_configuration(
                template,
                get_contract_tile_sizes(configuration, self.tile_dims),
            ),
            "",
        )


class BatchMmtTuner(DispatchTuner, BatchMmtParser):
    def get_transform_function_batch_mmt(
        self,
        problem_size: ProblemSize,
        functionName: str,
        configuration: BaseConfiguration,
    ) -> str:
        tile_sizes = get_batch_mmt_tile_sizes(configuration)
        config_mlir = configuration.get_mlir_config(tile_sizes)

        return f"""
transform.named_sequence @{functionName}(%generic: !transform.any_op {{transform.readonly}}) -> (!transform.any_op, !transform.any_param) {{
%mmt = transform.include @match_batch_mmt_i8_i8_i32 failures(propagate) (%generic) : (!transform.any_op) -> !transform.any_op
%lhs = transform.get_operand %generic[0] : (!transform.any_op) -> !transform.any_value
%rhs = transform.get_operand %generic[1] : (!transform.any_op) -> !transform.any_value
transform.iree.match.cast_compatible_type %lhs = tensor<{problem_size.lhs_type}> : !transform.any_value
transform.iree.match.cast_compatible_type %rhs = tensor<{problem_size.rhs_type}> : !transform.any_value
{config_mlir}
transform.yield %generic, %config : !transform.any_op, !transform.any_param
}}
"""

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: BaseConfiguration,
    ) -> MLIRTransformation:
        M, N, K = problem_size.MNK
        B = problem_size.matmul_size.B
        modified = indent(
            self.get_transform_function_batch_mmt(
                problem_size, f"match_batch_mmt_{B}x{M}x{N}x{K}", configuration
            ),
            "//   ",
        )
        modified += configuration.apply_configuration(
            template, get_batch_mmt_tile_sizes(configuration)
        )

        embeddable = indent(
            self.get_transform_function_batch_mmt(
                problem_size, f"match_op", configuration
            ),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)


class BatchMatmulTuner(DispatchTuner, BatchMatmulParser):
    def get_transform_function_batch_matmul(
        self,
        problem_size: ProblemSize,
        tile_dims: str,
        functionName: str,
        configuration: BaseConfiguration,
    ) -> str:
        input0 = f"tensor<{problem_size.lhs_type}>"
        input1 = f"tensor<{problem_size.rhs_type}>"
        output = f"tensor<{problem_size.res_type}>"

        tile_sizes = get_contract_tile_sizes(configuration, tile_dims)
        config_mlir = configuration.get_mlir_config(tile_sizes)

        return f"""
    transform.named_sequence @{functionName}(%batch_matmul: !transform.any_op {{transform.readonly}})
    -> (!transform.any_op, !transform.any_param) {{
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %batch_matmul {{
    ^bb0(%lhs: {input0}, %rhs: {input1}, %out: {output}):
        %13 = linalg.batch_matmul
        ins(%lhs, %rhs : {input0}, {input1})
        outs(%out : {output}) -> {output}
    }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
        {config_mlir}
    transform.yield %batch_matmul, %config : !transform.any_op, !transform.any_param
    }}
    """

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: BaseConfiguration,
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
        modified += configuration.apply_configuration(
            template,
            get_contract_tile_sizes(configuration, self.tile_dims),
        )

        embeddable = indent(
            self.get_transform_function_batch_matmul(
                problem_size, self.tile_dims, f"match_op", configuration
            ),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)


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
        mlir_module = parse_mlir(mlir_text, tuner_context)
        # Save the input file as the first candidate.
        with open(path.join(output, f"0.mlir"), "w") as f:
            f.write(mlir_text)

        dispatch_tuner_registry = DispatchTunerRegistry()
        dispatch_tuner_registry.register(
            [
                MmtTuner(),
                Mmt4dTuner(),
                ConvTuner(),
                ContractionTuner(lhs_dims, rhs_dims, tile_dims),
                BatchMmtTuner(),
                BatchMatmulTuner(lhs_dims, rhs_dims, tile_dims),
            ]
        )

        walk_result: OpWalkResult = walk_mlir_op(mlir_module, dispatch_tuner_registry)

        variant_op_list = iree_codegen.get_executable_variant_ops(mlir_module)
        assert len(variant_op_list) == 1, "Expect one executable variant op"
        variant_op = variant_op_list[0]
        # Get the MMA intrinisic intructions supported by the target.
        mma_list = iree_codegen.query_mma_intrinsics(variant_op)

        dispatch_tuner = walk_result.dispatch_tuner
        assert dispatch_tuner, "No suitable dispatch tuner found"
        problem_size: ProblemSize = dispatch_tuner.get_shapes(mlir_template)
        tune_logger.debug(str(problem_size))
        solutions_strategy = SolutionStrategyFactory.create_strategy(mlir_text, mma_list)
        configs = []
        for i, config in enumerate(
            solutions_strategy.generate_solutions(
                tune_logger, problem_size, num_subgroups
            )
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

        # TODO: Fix pickling for ir types.
        # with open(path.join(output, "configs.pkl"), "wb") as file:
        #    pickle.dump(configs, file)

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
