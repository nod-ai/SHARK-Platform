# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import candidate_gen

"""
Usage: python -m pytest test_tune.py
"""


def test_get_shaped_type_element_bitwidth():
    assert (
        candidate_gen.ShapedType([1024, 2048], candidate_gen.ElementType.i8).bitwidth
        == 8
    )
    assert (
        candidate_gen.ShapedType([2048], candidate_gen.ElementType.i32).bitwidth == 32
    )
    assert (
        candidate_gen.ShapedType(
            [2048, 512, 384], candidate_gen.ElementType.f8
        ).bitwidth
        == 8
    )
    assert (
        candidate_gen.ShapedType([1, 1], candidate_gen.ElementType.f16).bitwidth == 16
    )


def test_get_shaped_type_to_str():
    assert (
        str(candidate_gen.ShapedType([1024, 2048], candidate_gen.ElementType.i8))
        == "1024x2048xi8"
    )
    assert (
        str(candidate_gen.ShapedType([1024], candidate_gen.ElementType.f32))
        == "1024xf32"
    )
    assert (
        str(candidate_gen.ShapedType([1, 2, 3], candidate_gen.ElementType.f16))
        == "1x2x3xf16"
    )
    assert (
        str(candidate_gen.ShapedType([-1, 2, 3], candidate_gen.ElementType.f16))
        == "?x2x3xf16"
    )


def test_parse_tensor_type():
    assert candidate_gen.parse_tensor_type(
        "tensor<1x2x3xf32>"
    ) == candidate_gen.ShapedType([1, 2, 3], candidate_gen.ElementType.f32)
    assert candidate_gen.parse_tensor_type(
        "tensor<123xi8>"
    ) == candidate_gen.ShapedType([123], candidate_gen.ElementType.i8)


def test_get_mmt_tile_sizes():
    config = candidate_gen.Configuration(
        subgroup_size=0,
        workgroup_size=[],
        intrinsic="",
        tile_sizes=[128, 320, 32],
        subgroup_m_count=0,
        subgroup_n_count=0,
        waves_per_eu=0,
    )
    assert candidate_gen.get_mmt_tile_sizes(config) == [128, 320, 32]


def test_get_conv_tile_sizes():
    config = candidate_gen.Configuration(
        subgroup_size=64,
        workgroup_size=[256, 1, 1],
        intrinsic="#iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>",
        tile_sizes=[464, 320, 16],
        subgroup_m_count=1,
        subgroup_n_count=4,
        waves_per_eu=1,
    )
    assert candidate_gen.get_conv_tile_sizes(config) == [1, 1, 464, 320, 1, 1, 16]


def test_get_contract_tile_sizes():
    config = candidate_gen.Configuration(
        subgroup_size=32,
        workgroup_size=[16, 16, 1],
        intrinsic="",
        tile_sizes=[4, 8, 16],
        subgroup_m_count=1,
        subgroup_n_count=1,
        waves_per_eu=2,
    )
    assert candidate_gen.get_contract_tile_sizes(config, ["m", "n", "k"]) == [4, 8, 16]
    assert candidate_gen.get_contract_tile_sizes(config, ["n", "m", "k"]) == [8, 4, 16]
    assert candidate_gen.get_contract_tile_sizes(config, ["k", "n", "m"]) == [16, 8, 4]
    assert candidate_gen.get_contract_tile_sizes(config, ["k", "k", "k"]) == [
        16,
        16,
        16,
    ]


def test_get_pipeline_config():
    config1 = candidate_gen.Configuration(
        subgroup_size=32,
        workgroup_size=[16, 16, 1],
        intrinsic="",
        tile_sizes=[4, 8, 16],
        subgroup_m_count=1,
        subgroup_n_count=1,
        waves_per_eu=2,
    )
    config2 = candidate_gen.Configuration(
        subgroup_size=32,
        workgroup_size=[16, 16, 1],
        intrinsic="",
        tile_sizes=[4, 8, 16],
        subgroup_m_count=1,
        subgroup_n_count=1,
        waves_per_eu=4,
    )
    assert candidate_gen.get_pipeline_config(config1) == ", prefetch_shared_memory"
    assert (
        candidate_gen.get_pipeline_config(config2)
        == ', prefetch_shared_memory, llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}'
    )


def test_get_shapes_mmt():
    template = [
        r"%18 = tensor.empty() : tensor<2048x1280xf32>",
        r"%19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} ins(%cst : f32) outs(%18 : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>",
        r'%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %14 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%19 : tensor<2048x1280xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {',
        r"^bb0(%in: f16, %in_0: f16, %out: f32):",
    ]
    assert candidate_gen.get_shapes_mmt(template) == candidate_gen.ProblemSize(
        candidate_gen.MatmulSize(2048, 1280, 1280),
        candidate_gen.ShapedType([2048, 1280], candidate_gen.ElementType.f16),
        candidate_gen.ShapedType([1280, 1280], candidate_gen.ElementType.f16),
        candidate_gen.ShapedType([2048, 1280], candidate_gen.ElementType.f32),
        candidate_gen.DispatchKind.mmt,
    )


def test_get_shapes_conv():
    template = [
        r"%7 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 1, 1, 32]]>} ins(%cst : f32) outs(%4 : tensor<1x1x32x256xf32>) -> tensor<1x1x32x256xf32>",
        r"%8 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 1, 1, 32]]>, strides = dense<1> : vector<2xi64>} ins(%5, %6 : tensor<1x3x34x1280xf16>, tensor<3x3x1280x256xf16>) outs(%7 : tensor<1x1x32x256xf32>) -> tensor<1x1x32x256xf32>",
        r"flow.dispatch.tensor.store %8, %2, offsets = [%workgroup_id_z, %workgroup_id_y, 0, %3], sizes = [1, 1, 32, 256], strides = [1, 1, 1, 1] : tensor<1x1x32x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x32x32x1280xf32>>",
    ]
    assert candidate_gen.get_shapes_conv(template) == candidate_gen.ProblemSize(
        candidate_gen.MatmulSize(32, 256, 11520),
        candidate_gen.ShapedType([1, 3, 34, 1280], candidate_gen.ElementType.f16),
        candidate_gen.ShapedType([3, 3, 1280, 256], candidate_gen.ElementType.f16),
        candidate_gen.ShapedType([1, 1, 32, 256], candidate_gen.ElementType.f32),
        candidate_gen.DispatchKind.conv,
    )


def test_get_shapes_contract():
    template = [
        r"%18 = tensor.empty() : tensor<2048x1280xf32>",
        r"%19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} ins(%cst : f32) outs(%18 : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>",
        r'%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %14 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%19 : tensor<2048x1280xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {',
        r"^bb0(%in: f16, %in_0: f16, %out: f32):",
    ]
    assert candidate_gen.get_shapes_contract(
        template, "mk", "nk"
    ) == candidate_gen.ProblemSize(
        candidate_gen.MatmulSize(2048, 1280, 1280),
        candidate_gen.ShapedType([2048, 1280], candidate_gen.ElementType.f16),
        candidate_gen.ShapedType([1280, 1280], candidate_gen.ElementType.f16),
        candidate_gen.ShapedType([2048, 1280], candidate_gen.ElementType.f32),
        candidate_gen.DispatchKind.contraction,
    )


def test_get_shapes_batch_matmul():
    template = [
        "%10 = linalg.fill ins(%cst : f32) outs(%7 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>",
        "%11 = linalg.batch_matmul ins(%8, %9 : tensor<1x32x1024xf32>, tensor<1x1024x32xf32>) outs(%10 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>",
        "flow.dispatch.tensor.store %11, %2, offsets = [%arg0, %arg1, %arg2], sizes = [1, 32, 32], strides = [1, 1, 1] : tensor<1x32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x32x64xf32>>",
    ]
    assert candidate_gen.get_shapes_batch_matmul(
        template, "bmk", "bkn"
    ) == candidate_gen.ProblemSize(
        candidate_gen.MatmulSize(32, 32, 1024, 1),
        candidate_gen.ShapedType([1, 32, 1024], candidate_gen.ElementType.f32),
        candidate_gen.ShapedType([1, 1024, 32], candidate_gen.ElementType.f32),
        candidate_gen.ShapedType([1, 32, 32], candidate_gen.ElementType.f32),
        candidate_gen.DispatchKind.batch_matmul,
    )


def test_get_shapes_batch_mmt():
    template = [
        r"%19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 128, 128]]>} ins(%c0_i32 : i32) outs(%18 : tensor<2x4096x640xi32>) -> tensor<2x4096x640xi32>",
        r'%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%11, %12 : tensor<2x4096x640xi8>, tensor<2x640x640xi8>) outs(%19 : tensor<2x4096x640xi32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 128, 128]]>} {',
        r"flow.dispatch.tensor.store %21, %10, offsets = [0, 0, 0], sizes = [2, 4096, 640], strides = [1, 1, 1] : tensor<2x4096x640xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x4096x640xf16>>",
    ]
    assert candidate_gen.get_shapes_batch_mmt(template) == candidate_gen.ProblemSize(
        candidate_gen.MatmulSize(4096, 640, 640, 2),
        candidate_gen.ShapedType([2, 4096, 640], candidate_gen.ElementType.i8),
        candidate_gen.ShapedType([2, 640, 640], candidate_gen.ElementType.i8),
        candidate_gen.ShapedType([2, 4096, 640], candidate_gen.ElementType.i32),
        candidate_gen.DispatchKind.batch_mmt,
    )


def test_mfma_intrinsic_to_str():
    assert (
        str(candidate_gen.MfmaIntrinsic.mfma_f16_16x16x16_f32())
        == "MFMA_F16_16x16x16_F32"
    )
    assert (
        str(candidate_gen.MfmaIntrinsic.mfma_i8_32x32x16_i32())
        == "MFMA_I8_32x32x16_I32"
    )


def test_get_compatible_mfma_intrinsics():
    assert candidate_gen.get_compatible_mfma_intrinsics(
        candidate_gen.ProblemSize(
            candidate_gen.MatmulSize(2048, 1280, 1280),
            candidate_gen.ShapedType([2048, 1280], candidate_gen.ElementType.f16),
            candidate_gen.ShapedType([1280, 1280], candidate_gen.ElementType.f16),
            candidate_gen.ShapedType([2048, 1280], candidate_gen.ElementType.f32),
            candidate_gen.DispatchKind.mmt,
        )
    ) == [
        candidate_gen.MfmaIntrinsic.mfma_f16_16x16x16_f32(),
        candidate_gen.MfmaIntrinsic.mfma_f16_32x32x8_f32(),
    ]

    assert candidate_gen.get_compatible_mfma_intrinsics(
        candidate_gen.ProblemSize(
            candidate_gen.MatmulSize(2048, 1280, 1280),
            candidate_gen.ShapedType([2048, 1280], candidate_gen.ElementType.i8),
            candidate_gen.ShapedType([1280, 1280], candidate_gen.ElementType.i8),
            candidate_gen.ShapedType([2048, 1280], candidate_gen.ElementType.i32),
            candidate_gen.DispatchKind.mmt,
        )
    ) == [
        candidate_gen.MfmaIntrinsic.mfma_i8_16x16x32_i32(),
        candidate_gen.MfmaIntrinsic.mfma_i8_32x32x16_i32(),
    ]

    assert candidate_gen.get_compatible_mfma_intrinsics(
        candidate_gen.ProblemSize(
            candidate_gen.MatmulSize(968, 320, 640, 64),
            candidate_gen.ShapedType([64, 968, 640], candidate_gen.ElementType.f32),
            candidate_gen.ShapedType([64, 640, 320], candidate_gen.ElementType.f32),
            candidate_gen.ShapedType([64, 968, 320], candidate_gen.ElementType.f32),
            candidate_gen.DispatchKind.batch_matmul,
        )
    ) == [
        candidate_gen.MfmaIntrinsic.mfma_f16_16x16x16_f32(),
        candidate_gen.MfmaIntrinsic.mfma_f16_32x32x8_f32(),
    ]


def test_generate_solutions():
    matmul_size = candidate_gen.MatmulSize(2048, 3840, 1280)
    lhs_type = candidate_gen.ShapedType([2048, 1280], candidate_gen.ElementType.f16)
    rhs_type = candidate_gen.ShapedType([3840, 1280], candidate_gen.ElementType.f16)
    res_type = candidate_gen.ShapedType([2048, 3840], candidate_gen.ElementType.f32)
    problem_size = candidate_gen.ProblemSize(
        matmul_size, lhs_type, rhs_type, res_type, candidate_gen.DispatchKind.mmt
    )
    configs = candidate_gen.generate_solutions(problem_size, 4)
    assert configs is not None


def test_calculate_shared_memory_usage_in_bytes():
    matmul_size = candidate_gen.MatmulSize(1024, 1024, 1024)
    lhs_type = candidate_gen.ShapedType([1024, 1024], candidate_gen.ElementType.f16)
    rhs_type = candidate_gen.ShapedType([1024, 1024], candidate_gen.ElementType.f16)
    res_type = candidate_gen.ShapedType([1024, 1024], candidate_gen.ElementType.f32)
    problem_size = candidate_gen.ProblemSize(
        matmul_size, lhs_type, rhs_type, res_type, candidate_gen.DispatchKind.mmt
    )
    assert (
        candidate_gen.calculate_shared_memory_usage_in_bytes(problem_size, 512, 64, 128)
        == 147456
    )

    lhs_type = candidate_gen.ShapedType([1024, 1024], candidate_gen.ElementType.i8)
    problem_size = candidate_gen.ProblemSize(
        matmul_size, lhs_type, rhs_type, res_type, candidate_gen.DispatchKind.mmt
    )
    assert (
        candidate_gen.calculate_shared_memory_usage_in_bytes(problem_size, 512, 64, 128)
        == 81920
    )

    rhs_type = candidate_gen.ShapedType([1024, 1024], candidate_gen.ElementType.i32)
    problem_size = candidate_gen.ProblemSize(
        matmul_size, lhs_type, rhs_type, res_type, candidate_gen.DispatchKind.mmt
    )
    assert (
        candidate_gen.calculate_shared_memory_usage_in_bytes(problem_size, 128, 64, 32)
        == 12288
    )


def test_generate_constraints_valid_input():
    matmul_size = candidate_gen.MatmulSize(1024, 1024, 1024)
    lhs_type = candidate_gen.ShapedType([1024, 1024], candidate_gen.ElementType.f16)
    rhs_type = candidate_gen.ShapedType([1024, 1024], candidate_gen.ElementType.f16)
    res_type = candidate_gen.ShapedType([1024, 1024], candidate_gen.ElementType.f32)
    problem_size = candidate_gen.ProblemSize(
        matmul_size, lhs_type, rhs_type, res_type, candidate_gen.DispatchKind.mmt
    )
    # Define input parameters as z3 Ints
    m, n, k = (
        candidate_gen.z3.Int("m"),
        candidate_gen.z3.Int("n"),
        candidate_gen.z3.Int("k"),
    )
    subgroup_size = candidate_gen.z3.Int("subgroup_size")
    intrinsic_mn = candidate_gen.z3.Int("intrinsic_mn")
    intrinsic_k = candidate_gen.z3.Int("intrinsic_k")
    wg_x, wg_y, wg_z = (
        candidate_gen.z3.Int("wg_x"),
        candidate_gen.z3.Int("wg_y"),
        candidate_gen.z3.Int("wg_z"),
    )
    sg_m_cnt = candidate_gen.z3.Int("sg_m_cnt")
    sg_n_cnt = candidate_gen.z3.Int("sg_n_cnt")
    waves_per_eu = candidate_gen.z3.Int("waves_per_eu")

    constraints = candidate_gen.generate_constraints(
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

    solver = candidate_gen.z3.Solver()
    solver.add(constraints)

    # Check if the constraints are satisfiable
    assert solver.check() == candidate_gen.z3.sat


def test_generate_constraints_invalid_input():
    # Define input parameters that should lead to unsatisfiable constraints
    matmul_size = candidate_gen.MatmulSize(1024, 1024, 1024)
    lhs_type = candidate_gen.ShapedType([1024, 1024], candidate_gen.ElementType.f16)
    rhs_type = candidate_gen.ShapedType([1024, 1024], candidate_gen.ElementType.f16)
    res_type = candidate_gen.ShapedType([1024, 1024], candidate_gen.ElementType.f32)
    problem_size = candidate_gen.ProblemSize(
        matmul_size, lhs_type, rhs_type, res_type, candidate_gen.DispatchKind.mmt
    )
    m, n, k = (
        candidate_gen.z3.Int("m"),
        candidate_gen.z3.Int("n"),
        candidate_gen.z3.Int("k"),
    )
    subgroup_size = candidate_gen.z3.Int("subgroup_size")
    intrinsic_mn = candidate_gen.z3.Int("intrinsic_mn")
    intrinsic_k = candidate_gen.z3.Int("intrinsic_k")
    wg_x, wg_y, wg_z = (
        candidate_gen.z3.Int("wg_x"),
        candidate_gen.z3.Int("wg_y"),
        candidate_gen.z3.Int("wg_z"),
    )
    sg_m_cnt = candidate_gen.z3.Int("sg_m_cnt")
    sg_n_cnt = candidate_gen.z3.Int("sg_n_cnt")
    waves_per_eu = candidate_gen.z3.Int("waves_per_eu")

    constraints = candidate_gen.generate_constraints(
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

    solver = candidate_gen.z3.Solver()
    solver.add(constraints)

    # Check if the constraints are unsatisfiable
    assert solver.check() == candidate_gen.z3.unsat


def test_apply_params_mmt():
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>, subgroup_m_count = 16, subgroup_n_count = 16>",
        "<LLVMGPUVectorDistribute workgroup_size = [16, 16] subgroup_size = 16,",
        "<tile_sizes = [[8, 8, 8]]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}',
    ]

    M, N, K = 2048, 1280, 1280

    config = candidate_gen.Configuration(
        subgroup_size=16,
        workgroup_size=[16, 16, 1],
        intrinsic=candidate_gen.MfmaIntrinsic.mfma_f16_16x16x16_f32(),
        tile_sizes=[8, 8, 8],
        subgroup_m_count=16,
        subgroup_n_count=16,
        waves_per_eu=8,
    )

    problem_size = candidate_gen.ProblemSize(
        candidate_gen.MatmulSize(M, N, K),
        candidate_gen.ShapedType([M, K], candidate_gen.ElementType.f16),
        candidate_gen.ShapedType([N, K], candidate_gen.ElementType.f16),
        candidate_gen.ShapedType([M, N], candidate_gen.ElementType.f32),
        candidate_gen.DispatchKind.mmt,
    )
    modified, embeddable = candidate_gen.apply_params_mmt(
        problem_size, mlir_template, config
    )

    assert modified
    assert embeddable
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 16, subgroup_n_count = 16"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [16, 16, 1] subgroup_size = 16"
        in modified
    )
    assert "tile_sizes = [[8, 8, 8]]" in modified
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "8"}' in modified


def test_apply_params_conv():
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 16, subgroup_n_count = 16>",
        "<LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 1, 64, 128, 1, 1, 32]]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}',
    ]

    n, oh, ow, oc, fh, fw, ic = 2, 64, 64, 640, 3, 3, 640

    config = candidate_gen.Configuration(
        subgroup_size=64,
        workgroup_size=[256, 1, 1],
        intrinsic=candidate_gen.MfmaIntrinsic.mfma_f16_16x16x16_f32(),
        tile_sizes=[464, 320, 16],
        subgroup_m_count=1,
        subgroup_n_count=4,
        waves_per_eu=2,
    )

    problem_size = candidate_gen.ProblemSize(
        candidate_gen.MatmulSize(oh * ow, oc, fh * fw * ic),
        candidate_gen.ShapedType(
            [n, oh + 2, ow + 2, oc], candidate_gen.ElementType.f16
        ),
        candidate_gen.ShapedType([fh, fw, ic, oc], candidate_gen.ElementType.f16),
        candidate_gen.ShapedType([n, oh, ow, oc], candidate_gen.ElementType.f32),
        candidate_gen.DispatchKind.conv,
    )
    modified, embeddable = candidate_gen.apply_params_conv(
        problem_size, mlir_template, config
    )

    assert modified
    assert embeddable
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 1, subgroup_n_count = 4"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64"
        in modified
    )
    assert "tile_sizes = [[1, 1, 464, 320, 1, 1, 16]]" in modified
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}' in modified


def test_apply_params_contract():
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 1, 1, 64, 64, 128]]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}',
    ]

    tile_dims = "*mnk"
    problem_size = candidate_gen.ProblemSize(
        candidate_gen.MatmulSize(2048, 3840, 1280),
        candidate_gen.ShapedType([2, 1024, 1280], candidate_gen.ElementType.f16),
        candidate_gen.ShapedType([3, 20, 64, 1280], candidate_gen.ElementType.f16),
        candidate_gen.ShapedType([3, 2, 20, 1024, 64], candidate_gen.ElementType.f32),
        candidate_gen.DispatchKind.contraction,
    )

    config = candidate_gen.Configuration(
        subgroup_size=64,
        workgroup_size=[256, 1, 1],
        intrinsic=candidate_gen.MfmaIntrinsic.mfma_f16_32x32x8_f32(),
        tile_sizes=[480, 384, 32],
        subgroup_m_count=1,
        subgroup_n_count=4,
        waves_per_eu=2,
    )

    new_mlir, _embeddable = candidate_gen.apply_params_contract(
        problem_size, tile_dims, mlir_template, config
    )

    assert new_mlir
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>, subgroup_m_count = 1, subgroup_n_count = 4"
        in new_mlir
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64"
        in new_mlir
    )
    assert "tile_sizes = [[1, 480, 384, 32]]" in new_mlir
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}' in new_mlir


def test_apply_params_batch_matmul():
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 4, subgroup_n_count = 1>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [64, 4, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 128, 64, 64]]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}',
    ]

    tile_dims = "bmnk"
    problem_size = candidate_gen.ProblemSize(
        candidate_gen.MatmulSize(968, 320, 640, 64),
        candidate_gen.ShapedType([64, 968, 640], candidate_gen.ElementType.f16),
        candidate_gen.ShapedType([64, 640, 320], candidate_gen.ElementType.f16),
        candidate_gen.ShapedType([64, 968, 320], candidate_gen.ElementType.f32),
        candidate_gen.DispatchKind.batch_matmul,
    )

    config = candidate_gen.Configuration(
        subgroup_size=64,
        workgroup_size=[128, 2, 1],
        intrinsic=candidate_gen.MfmaIntrinsic.mfma_f16_32x32x8_f32(),
        tile_sizes=[416, 320, 128],
        subgroup_m_count=2,
        subgroup_n_count=2,
        waves_per_eu=2,
    )

    modified, embeddable = candidate_gen.apply_params_batch_matmul(
        problem_size, tile_dims, mlir_template, config
    )

    assert modified
    assert embeddable
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>, subgroup_m_count = 2, subgroup_n_count = 2"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64"
        in modified
    )
    assert "tile_sizes = [[1, 416, 320, 128]]" in modified
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}' in modified


def test_apply_params_batch_mmt_float():
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 4, subgroup_n_count = 1>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [64, 4, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 128, 128, 64]]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}',
    ]

    problem_size = candidate_gen.ProblemSize(
        candidate_gen.MatmulSize(4096, 640, 640, 2),
        candidate_gen.ShapedType([2, 4096, 640], candidate_gen.ElementType.f16),
        candidate_gen.ShapedType([2, 640, 640], candidate_gen.ElementType.f16),
        candidate_gen.ShapedType([2, 4096, 640], candidate_gen.ElementType.f32),
        candidate_gen.DispatchKind.batch_mmt,
    )

    config = candidate_gen.Configuration(
        subgroup_size=64,
        workgroup_size=[128, 2, 1],
        intrinsic=candidate_gen.MfmaIntrinsic.mfma_f16_16x16x16_f32(),
        tile_sizes=[128, 64, 128],
        subgroup_m_count=2,
        subgroup_n_count=2,
        waves_per_eu=2,
    )

    modified, embeddable = candidate_gen.apply_params_batch_mmt(
        problem_size, mlir_template, config
    )

    assert embeddable
    assert modified
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64"
        in modified
    )
    assert "tile_sizes = [[1, 128, 64, 128]]" in modified
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}' in modified


def test_apply_params_batch_mmt_int():
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_I8_16x16x32_I32>, subgroup_m_count = 4, subgroup_n_count = 1>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [64, 4, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 128, 128, 64]]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}',
    ]

    problem_size = candidate_gen.ProblemSize(
        candidate_gen.MatmulSize(4096, 640, 640, 2),
        candidate_gen.ShapedType([2, 4096, 640], candidate_gen.ElementType.i8),
        candidate_gen.ShapedType([2, 640, 640], candidate_gen.ElementType.i8),
        candidate_gen.ShapedType([2, 4096, 640], candidate_gen.ElementType.i32),
        candidate_gen.DispatchKind.batch_mmt,
    )

    config = candidate_gen.Configuration(
        subgroup_size=64,
        workgroup_size=[128, 2, 1],
        intrinsic=candidate_gen.MfmaIntrinsic.mfma_i8_32x32x16_i32(),
        tile_sizes=[128, 64, 128],
        subgroup_m_count=2,
        subgroup_n_count=2,
        waves_per_eu=4,
    )

    modified, embeddable = candidate_gen.apply_params_batch_mmt(
        problem_size, mlir_template, config
    )

    assert modified
    assert "//   transform.named_sequence @match_batch_mmt_2x4096x640x640(" in modified
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_I8_32x32x16_I32>, subgroup_m_count = 2, subgroup_n_count = 2"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64"
        in modified
    )
    assert "tile_sizes = [[1, 128, 64, 128]]" in modified
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}' in modified

    assert embeddable
    assert "transform.named_sequence @match_op(" in embeddable
    assert (
        "transform.include @match_batch_mmt_i8_i8_i32 failures(propagate)" in embeddable
    )
    assert (
        "transform.iree.match.cast_compatible_type %lhs = tensor<2x4096x640xi8> : !transform.any_value"
        in embeddable
    )
    assert (
        "transform.iree.match.cast_compatible_type %rhs = tensor<2x640x640xi8> : !transform.any_value"
        in embeddable
    )
    assert (
        "%config = transform.param.constant #iree_codegen.compilation_info<"
        in embeddable
    )
    assert "tile_sizes = [[1, 128, 64, 128]]" in embeddable
    assert 'llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}' in embeddable
    assert "workgroup_size = [128, 2, 1] subgroup_size = 64" in embeddable


def test_apply_params_broadcast_rhs_mmt():
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_I8_16x16x32_I32>, subgroup_m_count = 4, subgroup_n_count = 1>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [64, 4, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 128, 128, 64]]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}',
    ]

    problem_size = candidate_gen.ProblemSize(
        candidate_gen.MatmulSize(4096, 640, 640, 2),
        candidate_gen.ShapedType([2, 4096, 640], candidate_gen.ElementType.i8),
        candidate_gen.ShapedType([640, 640], candidate_gen.ElementType.i8),
        candidate_gen.ShapedType([2, 4096, 640], candidate_gen.ElementType.i32),
        candidate_gen.DispatchKind.broadcast_rhs_mmt,
    )

    config = candidate_gen.Configuration(
        subgroup_size=64,
        workgroup_size=[128, 2, 1],
        intrinsic=candidate_gen.MfmaIntrinsic.mfma_i8_32x32x16_i32(),
        tile_sizes=[128, 64, 128],
        subgroup_m_count=2,
        subgroup_n_count=2,
        waves_per_eu=4,
    )

    modified, embeddable = candidate_gen.apply_params_broadcast_rhs_mmt(
        problem_size, mlir_template, config
    )

    assert modified
    assert (
        "//   transform.named_sequence @match_broadcast_rhs_mmt_Bx4096x640x640("
        in modified
    )
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_I8_32x32x16_I32>, subgroup_m_count = 2, subgroup_n_count = 2"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64"
        in modified
    )
    assert "tile_sizes = [[1, 128, 64, 128]]" in modified
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}' in modified

    assert embeddable
    assert "transform.named_sequence @match_op(" in embeddable
    assert (
        "transform.include @match_broadcast_rhs_mmt_i8_i8_i32 failures(propagate)"
        in embeddable
    )
    assert (
        "transform.iree.match.cast_compatible_type %lhs = tensor<?x4096x640xi8> : !transform.any_value"
        in embeddable
    )
    assert (
        "transform.iree.match.cast_compatible_type %rhs = tensor<640x640xi8> : !transform.any_value"
        in embeddable
    )
    assert (
        "%config = transform.param.constant #iree_codegen.compilation_info<"
        in embeddable
    )
    assert "tile_sizes = [[1, 128, 64, 128]]" in embeddable
    assert 'llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}' in embeddable
    assert "workgroup_size = [128, 2, 1] subgroup_size = 64" in embeddable


def test_detect_broadcast_rhs_mmt():
    mlir_lines = [
        r"%18 = tensor.empty() : tensor<2x1024x10240xi32>",
        r"%19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 128, 128]]>} ins(%c0_i32 : i32) outs(%18 : tensor<2x1024x10240xi32>) -> tensor<2x1024x10240xi32>",
        r'%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%11, %12 : tensor<2x1024x1280xi8>, tensor<10240x1280xi8>) outs(%19 : tensor<2x1024x10240xi32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 128, 128]]>} {',
    ]
    assert candidate_gen.is_broadcast_rhs_mmt(mlir_lines)


def test_parse_mlir():
    mlir_str = r"""
    builtin.module  {
      func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
        %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    }
  """
    mlir_module = candidate_gen.parse_mlir(mlir_str)
    assert mlir_module != None
    assert isinstance(mlir_module, candidate_gen.ireec._mlir_libs._mlir.ir.Module)
    assert isinstance(
        mlir_module.body.operations[0], candidate_gen.ireec.dialects.func.FuncOp
    )
