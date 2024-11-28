# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest candidate_gen_test.py
"""

import pytest

from typing import Generator

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_gpu  # type: ignore

from . import candidate_gen
from . import common


@pytest.fixture
def tuner_ctx() -> Generator[common.TunerContext, None, None]:
    from logging import Logger
    from unittest.mock import MagicMock

    with ir.Context() as ctx:
        logger: Logger = MagicMock(spec=Logger)
        yield common.TunerContext(ctx, logger)


def remove_comments(mlir: str) -> str:
    return "\n".join(
        filter(lambda x: not x.lstrip().startswith("//"), mlir.splitlines())
    )


def test_apply_params_mmt(tuner_ctx: common.TunerContext) -> None:
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>, subgroup_m_count = 16, subgroup_n_count = 16>",
        "<LLVMGPUVectorDistribute workgroup_size = [16, 16] subgroup_size = 16,",
        "<tile_sizes = [[8, 8, 8]]>",
        "gpu_pipeline_options = #iree_gpu.pipeline_options<reorder_workgroups_strategy = None>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}',
    ]

    M, N, K = 2048, 1280, 1280

    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config_dict = {
        "mma_kind": mma_attr,
        "workgroup": ir.ArrayAttr.get(
            [
                ir.IntegerAttr.get(tuner_ctx.type.i32, 8),
                ir.IntegerAttr.get(tuner_ctx.type.i32, 8),
                ir.IntegerAttr.get(tuner_ctx.type.i32, 8),
            ]
        ),
        "reduction": ir.ArrayAttr.get([]),
        "subgroup_m_count": ir.IntegerAttr.get(tuner_ctx.type.i32, 16),
        "subgroup_n_count": ir.IntegerAttr.get(tuner_ctx.type.i32, 16),
    }

    lowering_config_attrs = ir.DictAttr.get(lowering_config_dict)
    config = common.Configuration(
        subgroup_size=16,
        workgroup_size=[16, 16, 1],
        lowering_config=iree_gpu.LoweringConfigAttr.get(lowering_config_attrs),
        gpu_pipeline_options=iree_gpu.PipelineOptionsAttr.get(
            prefetch_shared_memory=True
        ),
        waves_per_eu=8,
    )

    problem_size = common.ProblemSize(
        common.MatmulSize(M, N, K),
        common.ShapedType([M, K], tuner_ctx.type.f16),
        common.ShapedType([N, K], tuner_ctx.type.f16),
        common.ShapedType([M, N], tuner_ctx.type.f32),
        common.DispatchKind.mmt,
    )
    tf_mlir = candidate_gen.MmtTuner().apply_params(problem_size, mlir_template, config)

    modified = tf_mlir.modified
    embeddable = tf_mlir.embeddable

    assert modified
    modified = remove_comments(modified)
    assert embeddable
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 16, subgroup_n_count = 16"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [16, 16, 1] subgroup_size = 16"
        in modified
    )
    assert "tile_sizes = [[8, 8, 8]]" in modified
    assert (
        "gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>"
        in modified
    )
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "8"}' in modified


def test_apply_params_conv(tuner_ctx: common.TunerContext) -> None:
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 16, subgroup_n_count = 16>",
        "<LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 1, 64, 128, 1, 1, 32]]>",
        'gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}',
    ]

    n, oh, ow, oc, fh, fw, ic = 2, 64, 64, 640, 3, 3, 640

    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config_dict = {
        "mma_kind": mma_attr,
        "workgroup": ir.ArrayAttr.get(
            [
                ir.IntegerAttr.get(tuner_ctx.type.i32, 464),
                ir.IntegerAttr.get(tuner_ctx.type.i32, 320),
                ir.IntegerAttr.get(tuner_ctx.type.i32, 16),
            ]
        ),
        "reduction": ir.ArrayAttr.get([]),
        "subgroup_m_count": ir.IntegerAttr.get(tuner_ctx.type.i32, 1),
        "subgroup_n_count": ir.IntegerAttr.get(tuner_ctx.type.i32, 4),
    }
    lowering_config_attrs = ir.DictAttr.get(lowering_config_dict)
    config = common.Configuration(
        subgroup_size=64,
        workgroup_size=[256, 1, 1],
        lowering_config=iree_gpu.LoweringConfigAttr.get(lowering_config_attrs),
        gpu_pipeline_options=iree_gpu.PipelineOptionsAttr.get(
            reorder_workgroups_strategy=iree_gpu.ReorderWorkgroupsStrategyAttr.get(
                iree_gpu.ReorderWorkgroupsStrategy.Transpose
            )
        ),
        waves_per_eu=2,
    )

    problem_size = common.ProblemSize(
        common.MatmulSize(oh * ow, oc, fh * fw * ic),
        common.ShapedType([n, oh + 2, ow + 2, oc], tuner_ctx.type.f16),
        common.ShapedType([fh, fw, ic, oc], tuner_ctx.type.f16),
        common.ShapedType([n, oh, ow, oc], tuner_ctx.type.f32),
        common.DispatchKind.conv,
    )
    tf_mlir = candidate_gen.ConvTuner().apply_params(
        problem_size, mlir_template, config
    )

    modified = tf_mlir.modified
    embeddable = tf_mlir.embeddable

    assert modified
    modified = remove_comments(modified)
    assert embeddable
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 1, subgroup_n_count = 4"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64"
        in modified
    )
    assert "tile_sizes = [[1, 1, 464, 320, 1, 1, 16]]" in modified
    assert (
        "gpu_pipeline_options = #iree_gpu.pipeline_options<reorder_workgroups_strategy = <Transpose>>"
        in modified
    )
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}' in modified


def test_apply_params_contract(tuner_ctx: common.TunerContext) -> None:
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 2, subgroup_n_count = 2>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 1, 1, 64, 64, 128]]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}',
    ]

    tile_dims = "*mnk"
    problem_size = common.ProblemSize(
        common.MatmulSize(2048, 3840, 1280),
        common.ShapedType([2, 1024, 1280], tuner_ctx.type.f16),
        common.ShapedType([3, 20, 64, 1280], tuner_ctx.type.f16),
        common.ShapedType([3, 2, 20, 1024, 64], tuner_ctx.type.f32),
        common.DispatchKind.contraction,
    )

    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config_dict = {
        "mma_kind": mma_attr,
        "workgroup": ir.ArrayAttr.get(
            [
                ir.IntegerAttr.get(tuner_ctx.type.i32, 480),
                ir.IntegerAttr.get(tuner_ctx.type.i32, 384),
                ir.IntegerAttr.get(tuner_ctx.type.i32, 32),
            ]
        ),
        "reduction": ir.ArrayAttr.get([]),
        "subgroup_m_count": ir.IntegerAttr.get(tuner_ctx.type.i32, 1),
        "subgroup_n_count": ir.IntegerAttr.get(tuner_ctx.type.i32, 4),
    }

    lowering_config_attrs = ir.DictAttr.get(lowering_config_dict)
    config = common.Configuration(
        subgroup_size=64,
        workgroup_size=[256, 1, 1],
        lowering_config=iree_gpu.LoweringConfigAttr.get(lowering_config_attrs),
        gpu_pipeline_options=iree_gpu.PipelineOptionsAttr.get(),
        waves_per_eu=2,
    )

    tf_mlir = candidate_gen.ContractionTuner("mk", "nk", tile_dims).apply_params(
        problem_size, mlir_template, config
    )

    new_mlir = tf_mlir.modified

    assert new_mlir
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>, subgroup_m_count = 1, subgroup_n_count = 4"
        in new_mlir
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64"
        in new_mlir
    )
    assert "tile_sizes = [[1, 480, 384, 32]]" in new_mlir
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}' in new_mlir


def test_apply_params_batch_matmul(tuner_ctx: common.TunerContext) -> None:
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 4, subgroup_n_count = 1>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [64, 4, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 128, 64, 64]]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}',
    ]

    tile_dims = "bmnk"
    problem_size = common.ProblemSize(
        common.MatmulSize(968, 320, 640, 64),
        common.ShapedType([64, 968, 640], tuner_ctx.type.f16),
        common.ShapedType([64, 640, 320], tuner_ctx.type.f16),
        common.ShapedType([64, 968, 320], tuner_ctx.type.f32),
        common.DispatchKind.batch_matmul,
    )

    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config_dict = {
        "mma_kind": mma_attr,
        "workgroup": ir.ArrayAttr.get(
            [
                ir.IntegerAttr.get(tuner_ctx.type.i32, 416),
                ir.IntegerAttr.get(tuner_ctx.type.i32, 320),
                ir.IntegerAttr.get(tuner_ctx.type.i32, 128),
            ]
        ),
        "reduction": ir.ArrayAttr.get([]),
        "subgroup_m_count": ir.IntegerAttr.get(tuner_ctx.type.i32, 2),
        "subgroup_n_count": ir.IntegerAttr.get(tuner_ctx.type.i32, 2),
    }

    lowering_config_attrs = ir.DictAttr.get(lowering_config_dict)

    config = common.Configuration(
        subgroup_size=64,
        workgroup_size=[128, 2, 1],
        lowering_config=iree_gpu.LoweringConfigAttr.get(lowering_config_attrs),
        gpu_pipeline_options=iree_gpu.PipelineOptionsAttr.get(),
        waves_per_eu=2,
    )

    tf_mlir = candidate_gen.BatchMatmulTuner("mk", "nk", tile_dims).apply_params(
        problem_size, mlir_template, config
    )

    modified = tf_mlir.modified
    embeddable = tf_mlir.embeddable

    assert modified
    modified = remove_comments(modified)

    assert embeddable
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>, subgroup_m_count = 2, subgroup_n_count = 2"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64"
        in modified
    )
    assert "tile_sizes = [[1, 416, 320, 128]]" in modified
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}' in modified


def test_apply_params_batch_mmt_float(tuner_ctx: common.TunerContext) -> None:
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 4, subgroup_n_count = 1>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [64, 4, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 128, 128, 64]]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}',
    ]

    problem_size = common.ProblemSize(
        common.MatmulSize(4096, 640, 640, 2),
        common.ShapedType([2, 4096, 640], tuner_ctx.type.f16),
        common.ShapedType([2, 640, 640], tuner_ctx.type.f16),
        common.ShapedType([2, 4096, 640], tuner_ctx.type.f32),
        common.DispatchKind.batch_mmt,
    )

    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config_dict = {
        "mma_kind": mma_attr,
        "workgroup": ir.ArrayAttr.get(
            [
                ir.IntegerAttr.get(tuner_ctx.type.i32, 128),
                ir.IntegerAttr.get(tuner_ctx.type.i32, 64),
                ir.IntegerAttr.get(tuner_ctx.type.i32, 128),
            ]
        ),
        "reduction": ir.ArrayAttr.get([]),
        "subgroup_m_count": ir.IntegerAttr.get(tuner_ctx.type.i32, 2),
        "subgroup_n_count": ir.IntegerAttr.get(tuner_ctx.type.i32, 2),
    }

    lowering_config_attrs = ir.DictAttr.get(lowering_config_dict)
    config = common.Configuration(
        subgroup_size=64,
        workgroup_size=[128, 2, 1],
        lowering_config=iree_gpu.LoweringConfigAttr.get(lowering_config_attrs),
        gpu_pipeline_options=iree_gpu.PipelineOptionsAttr.get(),
        waves_per_eu=2,
    )

    tf_mlir = candidate_gen.BatchMmtTuner().apply_params(
        problem_size, mlir_template, config
    )

    modified = tf_mlir.modified
    embeddable = tf_mlir.embeddable

    assert embeddable
    assert modified
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 2, subgroup_n_count = 2"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64"
        in modified
    )
    assert "tile_sizes = [[1, 128, 64, 128]]" in modified
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}' in modified


def test_apply_params_batch_mmt_int(tuner_ctx: common.TunerContext) -> None:
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, subgroup_m_count = 4, subgroup_n_count = 1>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [64, 4, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 128, 128, 64]]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}',
    ]

    problem_size = common.ProblemSize(
        common.MatmulSize(4096, 640, 640, 2),
        common.ShapedType([2, 4096, 640], tuner_ctx.type.i8),
        common.ShapedType([2, 640, 640], tuner_ctx.type.i8),
        common.ShapedType([2, 4096, 640], tuner_ctx.type.i32),
        common.DispatchKind.batch_mmt,
    )

    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config_dict = {
        "mma_kind": mma_attr,
        "workgroup": ir.ArrayAttr.get(
            [
                ir.IntegerAttr.get(tuner_ctx.type.i32, 128),
                ir.IntegerAttr.get(tuner_ctx.type.i32, 64),
                ir.IntegerAttr.get(tuner_ctx.type.i32, 128),
            ]
        ),
        "reduction": ir.ArrayAttr.get([]),
        "subgroup_m_count": ir.IntegerAttr.get(tuner_ctx.type.i32, 2),
        "subgroup_n_count": ir.IntegerAttr.get(tuner_ctx.type.i32, 2),
    }

    lowering_config_attrs = ir.DictAttr.get(lowering_config_dict)
    config = common.Configuration(
        subgroup_size=64,
        workgroup_size=[128, 2, 1],
        lowering_config=iree_gpu.LoweringConfigAttr.get(lowering_config_attrs),
        gpu_pipeline_options=iree_gpu.PipelineOptionsAttr.get(),
        waves_per_eu=4,
    )

    tf_mlir = candidate_gen.BatchMmtTuner().apply_params(
        problem_size, mlir_template, config
    )

    modified = tf_mlir.modified
    embeddable = tf_mlir.embeddable

    assert modified
    assert "//   transform.named_sequence @match_batch_mmt_2x4096x640x640(" in modified
    modified = remove_comments(modified)

    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, subgroup_m_count = 2, subgroup_n_count = 2"
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


def test_apply_params_broadcast_rhs_mmt(tuner_ctx: common.TunerContext) -> None:
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, subgroup_m_count = 4, subgroup_n_count = 1>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [64, 4, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 128, 128, 64]]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}',
    ]

    problem_size = common.ProblemSize(
        common.MatmulSize(4096, 640, 640, 2),
        common.ShapedType([2, 4096, 640], tuner_ctx.type.i8),
        common.ShapedType([640, 640], tuner_ctx.type.i8),
        common.ShapedType([2, 4096, 640], tuner_ctx.type.i32),
        common.DispatchKind.broadcast_rhs_mmt,
    )

    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config_dict = {
        "mma_kind": mma_attr,
        "workgroup": ir.ArrayAttr.get(
            [
                ir.IntegerAttr.get(tuner_ctx.type.i32, 128),
                ir.IntegerAttr.get(tuner_ctx.type.i32, 64),
                ir.IntegerAttr.get(tuner_ctx.type.i32, 128),
            ]
        ),
        "reduction": ir.ArrayAttr.get([]),
        "subgroup_m_count": ir.IntegerAttr.get(tuner_ctx.type.i32, 2),
        "subgroup_n_count": ir.IntegerAttr.get(tuner_ctx.type.i32, 2),
    }

    lowering_config_attrs = ir.DictAttr.get(lowering_config_dict)
    config = common.Configuration(
        subgroup_size=64,
        workgroup_size=[128, 2, 1],
        lowering_config=iree_gpu.LoweringConfigAttr.get(lowering_config_attrs),
        gpu_pipeline_options=iree_gpu.PipelineOptionsAttr.get(),
        waves_per_eu=4,
    )

    tf_mlir = candidate_gen.ContractionTuner(
        "mk", "nk", "mnk"
    ).apply_params_broadcast_rhs_mmt(problem_size, mlir_template, config)

    modified = tf_mlir.modified
    embeddable = tf_mlir.embeddable

    assert modified
    assert (
        "//   transform.named_sequence @match_broadcast_rhs_mmt_Bx4096x640x640("
        in modified
    )
    modified = remove_comments(modified)

    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, subgroup_m_count = 2, subgroup_n_count = 2"
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


def test_detect_broadcast_rhs_mmt(tuner_ctx: common.TunerContext) -> None:
    mlir_lines = [
        r"%18 = tensor.empty() : tensor<2x1024x10240xi32>",
        r"%19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 128, 128]]>} ins(%c0_i32 : i32) outs(%18 : tensor<2x1024x10240xi32>) -> tensor<2x1024x10240xi32>",
        r'%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%11, %12 : tensor<2x1024x1280xi8>, tensor<10240x1280xi8>) outs(%19 : tensor<2x1024x10240xi32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 128, 128]]>} {',
    ]
    assert candidate_gen.ContractionTuner("mk", "nk", "mnk").is_broadcast_rhs_mmt(
        mlir_lines
    )
