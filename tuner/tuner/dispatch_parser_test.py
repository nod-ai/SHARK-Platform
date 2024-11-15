# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest candidate_gen_test.py
"""

import pytest

from logging import Logger
from unittest.mock import MagicMock

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import func  # type: ignore

from . import common
from . import dispatch_parser


def test_get_mmt_tile_sizes() -> None:
    config = dispatch_parser.Configuration(
        subgroup_size=0,
        workgroup_size=[],
        intrinsic=common.MfmaIntrinsic.mfma_f32_16x16x16_f16(),
        tile_sizes=[128, 320, 32],
        subgroup_m_count=0,
        subgroup_n_count=0,
        gpu_pipeline_options=common.GpuPipelineOptions(),
        waves_per_eu=0,
    )
    assert dispatch_parser.get_mmt_tile_sizes(config) == [128, 320, 32]


def test_get_conv_tile_sizes() -> None:
    config = dispatch_parser.Configuration(
        subgroup_size=64,
        workgroup_size=[256, 1, 1],
        intrinsic=common.MfmaIntrinsic.mfma_f32_16x16x16_f16(),
        tile_sizes=[464, 320, 16],
        subgroup_m_count=1,
        subgroup_n_count=4,
        gpu_pipeline_options=common.GpuPipelineOptions(),
        waves_per_eu=1,
    )
    assert dispatch_parser.ConvParser().get_conv_tile_sizes(config) == [
        1,
        1,
        464,
        320,
        1,
        1,
        16,
    ]


def test_get_contract_tile_sizes() -> None:
    config = dispatch_parser.Configuration(
        subgroup_size=32,
        workgroup_size=[16, 16, 1],
        intrinsic=common.MfmaIntrinsic.mfma_f32_16x16x16_f16(),
        tile_sizes=[4, 8, 16],
        subgroup_m_count=1,
        subgroup_n_count=1,
        gpu_pipeline_options=common.GpuPipelineOptions(),
        waves_per_eu=2,
    )
    assert dispatch_parser.get_contract_tile_sizes(config, "mnk") == [4, 8, 16]
    assert dispatch_parser.get_contract_tile_sizes(config, "nmk") == [8, 4, 16]
    assert dispatch_parser.get_contract_tile_sizes(config, "knm") == [16, 8, 4]
    assert dispatch_parser.get_contract_tile_sizes(config, "kkk") == [
        16,
        16,
        16,
    ]


def test_get_shapes_mmt() -> None:
    template = [
        r"%18 = tensor.empty() : tensor<2048x1280xf32>",
        r"%19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} ins(%cst : f32) outs(%18 : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>",
        r'%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %14 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%19 : tensor<2048x1280xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {',
        r"^bb0(%in: f16, %in_0: f16, %out: f32):",
    ]
    assert dispatch_parser.MmtParser().get_shapes(template) == common.ProblemSize(
        common.MatmulSize(2048, 1280, 1280),
        common.ShapedType([2048, 1280], common.ElementType.f16),
        common.ShapedType([1280, 1280], common.ElementType.f16),
        common.ShapedType([2048, 1280], common.ElementType.f32),
        dispatch_parser.DispatchKind.mmt,
    )


def test_get_shapes_conv() -> None:
    template = [
        r"%7 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 1, 1, 32]]>} ins(%cst : f32) outs(%4 : tensor<1x1x32x256xf32>) -> tensor<1x1x32x256xf32>",
        r"%8 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 1, 1, 32]]>, strides = dense<1> : vector<2xi64>} ins(%5, %6 : tensor<1x3x34x1280xf16>, tensor<3x3x1280x256xf16>) outs(%7 : tensor<1x1x32x256xf32>) -> tensor<1x1x32x256xf32>",
        r"flow.dispatch.tensor.store %8, %2, offsets = [%workgroup_id_z, %workgroup_id_y, 0, %3], sizes = [1, 1, 32, 256], strides = [1, 1, 1, 1] : tensor<1x1x32x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x32x32x1280xf32>>",
    ]
    assert dispatch_parser.ConvParser().get_shapes(template) == common.ProblemSize(
        common.MatmulSize(32, 256, 11520),
        common.ShapedType([1, 3, 34, 1280], common.ElementType.f16),
        common.ShapedType([3, 3, 1280, 256], common.ElementType.f16),
        common.ShapedType([1, 1, 32, 256], common.ElementType.f32),
        dispatch_parser.DispatchKind.conv,
    )


def test_get_shapes_contract() -> None:
    template = [
        r"%18 = tensor.empty() : tensor<2048x1280xf32>",
        r"%19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} ins(%cst : f32) outs(%18 : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>",
        r'%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %14 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%19 : tensor<2048x1280xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {',
        r"^bb0(%in: f16, %in_0: f16, %out: f32):",
    ]
    assert dispatch_parser.ContractionParser("mk", "nk", "mnk").get_shapes(
        template
    ) == common.ProblemSize(
        common.MatmulSize(2048, 1280, 1280),
        common.ShapedType([2048, 1280], common.ElementType.f16),
        common.ShapedType([1280, 1280], common.ElementType.f16),
        common.ShapedType([2048, 1280], common.ElementType.f32),
        dispatch_parser.DispatchKind.contraction,
    )


def test_get_shapes_batch_matmul() -> None:
    template = [
        "%10 = linalg.fill ins(%cst : f32) outs(%7 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>",
        "%11 = linalg.batch_matmul ins(%8, %9 : tensor<1x32x1024xf32>, tensor<1x1024x32xf32>) outs(%10 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>",
        "flow.dispatch.tensor.store %11, %2, offsets = [%arg0, %arg1, %arg2], sizes = [1, 32, 32], strides = [1, 1, 1] : tensor<1x32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x32x64xf32>>",
    ]
    assert dispatch_parser.BatchMatmulParser("bmk", "bkn", "mnk").get_shapes(
        template
    ) == common.ProblemSize(
        common.MatmulSize(32, 32, 1024, 1),
        common.ShapedType([1, 32, 1024], common.ElementType.f32),
        common.ShapedType([1, 1024, 32], common.ElementType.f32),
        common.ShapedType([1, 32, 32], common.ElementType.f32),
        dispatch_parser.DispatchKind.batch_matmul,
    )


def test_get_shapes_batch_mmt() -> None:
    template = [
        r"%19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 128, 128]]>} ins(%c0_i32 : i32) outs(%18 : tensor<2x4096x640xi32>) -> tensor<2x4096x640xi32>",
        r'%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%11, %12 : tensor<2x4096x640xi8>, tensor<2x640x640xi8>) outs(%19 : tensor<2x4096x640xi32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 128, 128]]>} {',
        r"flow.dispatch.tensor.store %21, %10, offsets = [0, 0, 0], sizes = [2, 4096, 640], strides = [1, 1, 1] : tensor<2x4096x640xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x4096x640xf16>>",
    ]
    assert dispatch_parser.BatchMmtParser().get_shapes(template) == common.ProblemSize(
        common.MatmulSize(4096, 640, 640, 2),
        common.ShapedType([2, 4096, 640], common.ElementType.i8),
        common.ShapedType([2, 640, 640], common.ElementType.i8),
        common.ShapedType([2, 4096, 640], common.ElementType.i32),
        dispatch_parser.DispatchKind.batch_mmt,
    )


def test_parse_mlir() -> None:
    with ir.Context() as ctx:
        mlir_str = r"""
        builtin.module  {
        func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
            %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
            return %0 : tensor<4xf32>
        }
        }
    """
        logger: Logger = MagicMock(spec=Logger)
        tuner_context = common.TunerContext(ctx, logger)
        mlir_module = dispatch_parser.parse_mlir(mlir_str, tuner_context)
        assert mlir_module is not None
        assert isinstance(mlir_module, ir.Module)
        assert isinstance(mlir_module.body.operations[0], func.FuncOp)
