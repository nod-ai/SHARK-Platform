# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest common_test.py
"""

import pytest
from . import common

from typing import Generator

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_gpu  # type: ignore


@pytest.fixture
def tuner_ctx() -> Generator[common.TunerContext, None, None]:
    from logging import Logger
    from unittest.mock import MagicMock

    with ir.Context() as ctx:
        logger: Logger = MagicMock(spec=Logger)
        yield common.TunerContext(ctx, logger)


@pytest.fixture
def mlir_ctx() -> Generator[ir.Context, None, None]:
    with ir.Context() as ctx:
        yield ctx


def test_get_shaped_type_element_bitwidth(tuner_ctx: common.TunerContext) -> None:
    assert common.ShapedType([1024, 2048], tuner_ctx.type.i8).bitwidth == 8
    assert common.ShapedType([2048], tuner_ctx.type.i32).bitwidth == 32
    assert common.ShapedType([2048, 512, 384], tuner_ctx.type.f8E4M3FNUZ).bitwidth == 8
    assert common.ShapedType([1, 1], tuner_ctx.type.f16).bitwidth == 16


def test_get_shaped_type_to_str(tuner_ctx: common.TunerContext) -> None:
    assert str(common.ShapedType([1024, 2048], tuner_ctx.type.i8)) == "1024x2048xi8"
    assert str(common.ShapedType([1024], tuner_ctx.type.f32)) == "1024xf32"
    assert str(common.ShapedType([1, 2, 3], tuner_ctx.type.f16)) == "1x2x3xf16"
    assert str(common.ShapedType([-1, 2, 3], tuner_ctx.type.f16)) == "?x2x3xf16"


def test_gpu_pipeline_options(tuner_ctx: common.TunerContext) -> None:
    options = iree_gpu.PipelineOptionsAttr.get()
    assert str(options) == "#iree_gpu.pipeline_options<>"

    options = iree_gpu.PipelineOptionsAttr.get(prefetch_shared_memory=True)
    assert str(options) == "#iree_gpu.pipeline_options<prefetch_shared_memory = true>"

    options = iree_gpu.PipelineOptionsAttr.get(
        prefetch_shared_memory=True, no_reduce_shared_memory_bank_conflicts=False
    )
    assert (
        str(options)
        == "#iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>"
    )

    options = iree_gpu.PipelineOptionsAttr.get(
        reorder_workgroups_strategy=iree_gpu.ReorderWorkgroupsStrategyAttr.get(
            iree_gpu.ReorderWorkgroupsStrategy.Transpose
        )
    )
    assert (
        str(options)
        == "#iree_gpu.pipeline_options<reorder_workgroups_strategy = <Transpose>>"
    )


def test_get_pipeline_config(tuner_ctx: common.TunerContext) -> None:
    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        mma_kind=mma_attr,
        workgroup=[4, 8, 0],
        reduction=[0, 0, 16],
        subgroup_m_count=1,
        subgroup_n_count=1,
    )
    config = common.Configuration(
        subgroup_size=32,
        workgroup_size=[16, 16, 1],
        lowering_config=lowering_config,
        gpu_pipeline_options=iree_gpu.PipelineOptionsAttr.get(),
        waves_per_eu=2,
    )
    config1_str: str = common.get_pipeline_config(config)
    assert config1_str == ""

    config.waves_per_eu = 4
    config2_str: str = common.get_pipeline_config(config)
    assert config2_str == ', llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}'

    config.gpu_pipeline_options = iree_gpu.PipelineOptionsAttr.get(
        prefetch_shared_memory=True
    )
    config3_str = common.get_pipeline_config(config)
    assert (
        config3_str
        == ', gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}'
    )


def test_get_compatible_mfma_intrinsics(tuner_ctx: common.TunerContext) -> None:
    assert common.get_compatible_mfma_intrinsics(
        common.ProblemSize(
            common.MatmulSize(2048, 1280, 1280),
            common.ShapedType([2048, 1280], tuner_ctx.type.f16),
            common.ShapedType([1280, 1280], tuner_ctx.type.f16),
            common.ShapedType([2048, 1280], tuner_ctx.type.f32),
            common.DispatchKind.mmt,
        ),
        [
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
        ],
    ) == [
        iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
        iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
    ]

    assert common.get_compatible_mfma_intrinsics(
        common.ProblemSize(
            common.MatmulSize(2048, 1280, 1280),
            common.ShapedType([2048, 1280], tuner_ctx.type.i8),
            common.ShapedType([1280, 1280], tuner_ctx.type.i8),
            common.ShapedType([2048, 1280], tuner_ctx.type.i32),
            common.DispatchKind.mmt,
        ),
        [
            iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
            iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
        ],
    ) == [
        iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
        iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
    ]

    assert common.get_compatible_mfma_intrinsics(
        common.ProblemSize(
            common.MatmulSize(968, 320, 640, 64),
            common.ShapedType([64, 968, 640], tuner_ctx.type.f32),
            common.ShapedType([64, 640, 320], tuner_ctx.type.f32),
            common.ShapedType([64, 968, 320], tuner_ctx.type.f32),
            common.DispatchKind.batch_matmul,
        ),
        [
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
        ],
    ) == [
        iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
        iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
    ]

    assert common.get_compatible_mfma_intrinsics(
        common.ProblemSize(
            common.MatmulSize(968, 320, 640, 64),
            common.ShapedType([64, 968, 640], tuner_ctx.type.f32),
            common.ShapedType([64, 640, 320], tuner_ctx.type.f32),
            common.ShapedType([64, 968, 320], tuner_ctx.type.f32),
            common.DispatchKind.batch_matmul,
        ),
        [
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
        ],
    ) == [
        iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
    ]

    assert (
        common.get_compatible_mfma_intrinsics(
            common.ProblemSize(
                common.MatmulSize(968, 320, 640, 64),
                common.ShapedType([64, 968, 640], tuner_ctx.type.f32),
                common.ShapedType([64, 640, 320], tuner_ctx.type.f32),
                common.ShapedType([64, 968, 320], tuner_ctx.type.f32),
                common.DispatchKind.batch_matmul,
            ),
            [
                iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
                iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
            ],
        )
        == []
    )


def test_get_lowering_config(tuner_ctx: common.TunerContext) -> None:
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        workgroup=[4, 8, 0],
        reduction=[0, 0, 16],
        subgroup_m_count=1,
        subgroup_n_count=1,
    )

    assert (
        str(lowering_config)
        == "#iree_gpu.lowering_config<{reduction = [0, 0, 16], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [4, 8, 0]}>"
    )

    config = common.Configuration(
        subgroup_size=32,
        workgroup_size=[16, 16, 1],
        lowering_config=lowering_config,
        gpu_pipeline_options=iree_gpu.PipelineOptionsAttr.get(),
        waves_per_eu=2,
    )

    assert common.get_intrinsic(config) is None
    assert common.get_subgroup_m_count(config) == 1
    assert common.get_subgroup_n_count(config) == 1
