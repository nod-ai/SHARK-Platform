# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from iree.runtime import (  # type: ignore
    HalElementType,
)

from shortfin.framework.session import DeviceSession
from shortfin.llm.config import (
    CacheParams,
    ModelParams,
    ServiceParams,
)

from shortfin.llm.service import (
    GenerateRequest,
    GenerateResponsePart,
)

from shortfin.llm.attn_block_cache import (
    create_attn_block_cache_module,
    AttnBlockCache,
)

from shortfin.llm.impl.service_v1 import (
    GenerateServiceV1,
)

from shortfin.llm.testing.fake_v1_module import (
    create_fake_module,
)


@pytest.fixture
def cache_params(model_params: ModelParams) -> CacheParams:
    return CacheParams(model=model_params, device_block_count=128, block_pos_stride=16)


@pytest.fixture
def model_params() -> ModelParams:
    return ModelParams(
        module_name="AwesomeLLM",
        module_abi_version=1,
        attn_dtype=HalElementType.FLOAT_16,
        max_seq_len=128,
        transformer_block_count=32,
        attn_head_count=32,
        attn_head_dim=128,
        block_seq_stride=16,
        prefill_batch_sizes=[1, 4, 16],
        decode_batch_sizes=[1, 4, 16],
    )


@pytest.fixture
def uninitialized_session(model_params: ModelParams):
    from iree.runtime._binding import disable_leak_checker  # type: ignore

    disable_leak_checker()
    session = DeviceSession(uri="local-task", queue_count=2)
    yield session
    session.shutdown()
    del session


@pytest.fixture
def attn_block_cache(
    uninitialized_session: DeviceSession, cache_params: CacheParams
) -> AttnBlockCache:
    return AttnBlockCache(uninitialized_session, cache_params)


@pytest.fixture
def session(
    model_params: ModelParams,
    uninitialized_session: DeviceSession,
    attn_block_cache: AttnBlockCache,
):
    session = uninitialized_session
    lms = session.create_module_set("AwesomeLLM", context_count=1)
    lms.add(
        create_attn_block_cache_module(attn_block_cache),
        create_fake_module(session.device, "AwesomeLLM", model_params=model_params),
    )
    lms.initialize()
    return session


@pytest.fixture
def service(
    session: DeviceSession,
    cache_params: CacheParams,
    model_params: ModelParams,
    attn_block_cache: AttnBlockCache,
):
    params = ServiceParams(cache=cache_params, model=model_params)
    return GenerateServiceV1(session=session, params=params, cache=attn_block_cache)


def test_single(service: GenerateServiceV1):
    state = service.start()

    async def task():
        await state.set_sequences(
            requests=[
                GenerateRequest(
                    "1",
                    "hello, tell me a story",
                    [3, 4, 5, 12, 23, 88, 10, 2, 5, 9, 12, 13, 99, 56, 33, 124, 73],
                ),
                GenerateRequest("2", "goodbye", [9, 10]),
            ]
        )
        guarded_outputs = await state.prefill()
        prefill_ids = await guarded_outputs.resolve(state.host_context)
        print(
            "PREFILL IDS:",
            prefill_ids,
            ":\n",
            prefill_ids.map().asarray(
                prefill_ids.shape, HalElementType.map_to_dtype(prefill_ids.element_type)
            ),
        )
        await state.recycle()

    state.host_context.run_sync(task())
