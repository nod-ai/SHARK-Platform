# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest candidate_gen_test.py
"""

import pytest
import z3  # type: ignore

from typing import Generator

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_gpu  # type: ignore

from . import common
from . import dispatch_constraints


@pytest.fixture
def tuner_ctx() -> Generator[common.TunerContext, None, None]:
    from logging import Logger
    from unittest.mock import MagicMock

    with ir.Context() as ctx:
        logger: Logger = MagicMock(spec=Logger)
        yield common.TunerContext(ctx, logger)


def test_generate_solutions_llvmgpu(tuner_ctx: common.TunerContext) -> None:
    matmul_size = common.MatmulSize(2048, 3840, 1280)
    lhs_type = common.ShapedType([2048, 1280], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([3840, 1280], tuner_ctx.type.f16)
    res_type = common.ShapedType([2048, 3840], tuner_ctx.type.f32)
    problem_size = common.ProblemSize(
        matmul_size, lhs_type, rhs_type, res_type, common.DispatchKind.mmt
    )
    gpu_strategy = dispatch_constraints.LLVMGPUSolutionStrategy([
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
            iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
            iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
        ],)
    configs = gpu_strategy.generate_solutions(
        tuner_ctx.logger,
        problem_size,
        4,
    )

    assert configs is not None


def test_generate_solutions_llvmcpu(tuner_ctx: common.TunerContext) -> None:
    matmul_size = common.MatmulSize(2048, 3840, 1280)
    lhs_type = common.ShapedType([2048, 1280], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([3840, 1280], tuner_ctx.type.f16)
    res_type = common.ShapedType([2048, 3840], tuner_ctx.type.f32)
    problem_size = common.ProblemSize(
        matmul_size, lhs_type, rhs_type, res_type, common.DispatchKind.mmt
    )
    cpu_strategy = dispatch_constraints.LLVMCPUSolutionStrategy()
    configs = cpu_strategy.generate_solutions(
        tuner_ctx.logger,
        problem_size,
        4,
    )

    assert configs is not None


def test_calculate_shared_memory_usage_in_bytes(tuner_ctx: common.TunerContext) -> None:
    matmul_size = common.MatmulSize(1024, 1024, 1024)
    lhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    res_type = common.ShapedType([1024, 1024], tuner_ctx.type.f32)
    problem_size = common.ProblemSize(
        matmul_size, lhs_type, rhs_type, res_type, common.DispatchKind.mmt
    )
    assert (
        dispatch_constraints.calculate_shared_memory_usage_in_bytes(
            problem_size, 512, 64, 128
        )
        == 147456
    )

    lhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.i8)
    problem_size = common.ProblemSize(
        matmul_size, lhs_type, rhs_type, res_type, common.DispatchKind.mmt
    )
    assert (
        dispatch_constraints.calculate_shared_memory_usage_in_bytes(
            problem_size, 512, 64, 128
        )
        == 81920
    )

    rhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.i32)
    problem_size = common.ProblemSize(
        matmul_size, lhs_type, rhs_type, res_type, common.DispatchKind.mmt
    )
    assert (
        dispatch_constraints.calculate_shared_memory_usage_in_bytes(
            problem_size, 128, 64, 32
        )
        == 12288
    )


def test_generate_constraints_valid_input_llvmgpu(tuner_ctx: common.TunerContext) -> None:
    matmul_size = common.MatmulSize(1024, 1024, 1024)
    lhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    res_type = common.ShapedType([1024, 1024], tuner_ctx.type.f32)
    problem_size = common.ProblemSize(
        matmul_size, lhs_type, rhs_type, res_type, common.DispatchKind.mmt
    )
    # Define input parameters as z3 Ints
    m, n, k = (
        dispatch_constraints.z3.Int("m"),
        z3.Int("n"),
        z3.Int("k"),
    )
    subgroup_size = z3.Int("subgroup_size")
    intrinsic_mn = z3.Int("intrinsic_mn")
    intrinsic_k = z3.Int("intrinsic_k")
    wg_x, wg_y, wg_z = (
        z3.Int("wg_x"),
        z3.Int("wg_y"),
        z3.Int("wg_z"),
    )
    sg_m_cnt = z3.Int("sg_m_cnt")
    sg_n_cnt = z3.Int("sg_n_cnt")
    waves_per_eu = z3.Int("waves_per_eu")

    solution_strategy = dispatch_constraints.LLVMGPUSolutionStrategy([
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
            iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
            iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
        ],)
    constraints = solution_strategy.generate_constraints(
        problem_size,
        [m, n, k],
        4,
        subgroup_size,
        [intrinsic_mn, intrinsic_k],
        [wg_x, wg_y, wg_z],
        sg_m_cnt,
        sg_n_cnt,
        waves_per_eu,
    )

    solver = z3.Solver()
    solver.add(constraints)

    # Check if the constraints are satisfiable
    assert solver.check() == z3.sat


def test_generate_constraints_valid_input_llvmcpu(tuner_ctx: common.TunerContext) -> None:
    matmul_size = common.MatmulSize(1024, 1024, 1024)
    lhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    res_type = common.ShapedType([1024, 1024], tuner_ctx.type.f32)
    problem_size = common.ProblemSize(
        matmul_size, lhs_type, rhs_type, res_type, common.DispatchKind.mmt
    )
    m, n, k, m0, n0, k0 = (
        z3.Int("m"),
        z3.Int("n"),
        z3.Int("k"),
        z3.Int("m0"),
        z3.Int("n0"),
        z3.Int("k0"),
    )

    solution_strategy = dispatch_constraints.LLVMCPUSolutionStrategy()
    constraints = solution_strategy.generate_constraints(
        problem_size,
        [m, n, k, m0, n0, k0],
    )

    solver = z3.Solver()
    solver.add(constraints)

    assert solver.check() == z3.sat


def test_generate_constraints_invalid_input_llvmgpu(tuner_ctx: common.TunerContext) -> None:
    # Define input parameters that should lead to unsatisfiable constraints
    matmul_size = common.MatmulSize(1024, 1024, 1024)
    lhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    res_type = common.ShapedType([1024, 1024], tuner_ctx.type.f32)
    problem_size = common.ProblemSize(
        matmul_size, lhs_type, rhs_type, res_type, common.DispatchKind.mmt
    )
    m, n, k = (
        z3.Int("m"),
        z3.Int("n"),
        z3.Int("k"),
    )
    subgroup_size = z3.Int("subgroup_size")
    intrinsic_mn = z3.Int("intrinsic_mn")
    intrinsic_k = z3.Int("intrinsic_k")
    wg_x, wg_y, wg_z = (
        z3.Int("wg_x"),
        z3.Int("wg_y"),
        z3.Int("wg_z"),
    )
    sg_m_cnt = z3.Int("sg_m_cnt")
    sg_n_cnt = z3.Int("sg_n_cnt")
    waves_per_eu = z3.Int("waves_per_eu")

    solution_strategy = dispatch_constraints.LLVMGPUSolutionStrategy([
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
            iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
            iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
        ],)
    constraints = solution_strategy.generate_constraints(
        problem_size,
        [m, n, k],
        4,
        subgroup_size,
        [intrinsic_mn, intrinsic_k],
        [wg_x, wg_y, wg_z],
        sg_m_cnt,
        sg_n_cnt,
        waves_per_eu,
    )
    constraints.append(m > 1000)  # Adding an additional unsatisfiable constraint

    solver = z3.Solver()
    solver.add(constraints)

    # Check if the constraints are unsatisfiable
    assert solver.check() == z3.unsat


def test_generate_constraints_invalid_input_llvmcpu(tuner_ctx: common.TunerContext) -> None:
    matmul_size = common.MatmulSize(1024, 1024, 1024)
    lhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    res_type = common.ShapedType([1024, 1024], tuner_ctx.type.f32)
    problem_size = common.ProblemSize(
        matmul_size, lhs_type, rhs_type, res_type, common.DispatchKind.mmt
    )
    m, n, k, m0, n0, k0 = (
        z3.Int("m"),
        z3.Int("n"),
        z3.Int("k"),
        z3.Int("m0"),
        z3.Int("n0"),
        z3.Int("k0"),
    )

    solution_strategy = dispatch_constraints.LLVMCPUSolutionStrategy()
    constraints = solution_strategy.generate_constraints(
        problem_size,
        [m, n, k, m0, n0, k0],
    )

    constraints.append(m > 1000)  # Adding an additional unsatisfiable constraint

    solver = z3.Solver()
    solver.add(constraints)

    # Check if the constraints are unsatisfiable
    assert solver.check() == z3.unsat
