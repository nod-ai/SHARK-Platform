# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest spec_builder_test.py
"""

import pytest

from typing import Generator

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen  # type: ignore

from . import spec_builder
from . import common


@pytest.fixture
def tuner_ctx() -> Generator[common.TunerContext, None, None]:
    from logging import Logger
    from unittest.mock import MagicMock

    with ir.Context() as ctx:
        logger: Logger = MagicMock(spec=Logger)
        yield common.TunerContext(ctx, logger)


def test_build_vector_distribute_translation_info(
    tuner_ctx: common.TunerContext,
) -> None:
    context = tuner_ctx.mlir_ctx
    subgroup_size = 16
    workgroup_size = [16, 16, 1]
    config = common.Configuration(
        subgroup_size=subgroup_size,
        workgroup_size=workgroup_size,
        intrinsic=common.MfmaIntrinsic.mfma_f32_16x16x16_f16(),
        tile_sizes=[8, 8, 8],
        subgroup_m_count=16,
        subgroup_n_count=16,
        gpu_pipeline_options=common.GpuPipelineOptions(prefetch_shared_memory=True),
        waves_per_eu=8,
    )
    translation_info = spec_builder.build_vector_distribute_translation_info(
        config,
        context,
    )

    vecdist_pipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    vecdist_pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        vecdist_pipeline, context
    )

    assert translation_info
    assert translation_info.pass_pipeline == vecdist_pipeline_attr
    assert translation_info.subgroup_size == subgroup_size
    assert translation_info.workgroup_size == workgroup_size
    assert (
        "gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>"
        in str(translation_info.configuration)
    )
    assert 'llvm_func_attrs = {"amdgpu-waves-per-eu" = "8"}' in str(
        translation_info.configuration
    )


def test_build_vector_distribute_lowering_config(
    tuner_ctx: common.TunerContext,
) -> None:
    context = tuner_ctx.mlir_ctx
    intrinsic = common.MfmaIntrinsic.mfma_f32_16x16x16_f16()
    subgroup_m_count = 16
    subgroup_n_count = 16
    config = common.Configuration(
        subgroup_size=16,
        workgroup_size=[16, 16, 1],
        intrinsic=intrinsic,
        tile_sizes=[8, 8, 8],
        subgroup_m_count=subgroup_m_count,
        subgroup_n_count=subgroup_n_count,
        gpu_pipeline_options=common.GpuPipelineOptions(),
        waves_per_eu=2,
    )
    workgroup_tile_sizes = [8, 8, 0]
    reduction_tile_sizes = [0, 0, 8]
    lowering_config = spec_builder.build_vector_distribute_lowering_config(
        config,
        reduction_tile_sizes,
        workgroup_tile_sizes,
        context,
    )

    assert lowering_config
    assert str(intrinsic) in str(lowering_config)
    assert "workgroup = [8, 8, 0]" in str(lowering_config)
    assert "reduction = [0, 0, 8]" in str(lowering_config)
    assert f"subgroup_m_count = {subgroup_m_count}" in str(lowering_config)
    assert f"subgroup_n_count = {subgroup_n_count}" in str(lowering_config)
