# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Implements the BatchGenerateService for V1 compiled models.

This is far from where we want to land but is intended for first round bootstrapping.
Perhaps the biggest issue is that it wouldn't mate well as-is with samplers.
"""

import asyncio
from dataclasses import dataclass

import numpy as np

from iree.runtime import (  # type: ignore
    HalBufferView,
    HalCommandBuffer,
    HalElementType,
    HalFence,
    VmFunction,
    VmVariantList,
)

from ...framework.logging import get_logger, NDEBUG
from ...framework.session import (
    AsyncResources,
    DeviceSession,
    TimelineGuarded,
    TransferBufferPool,
    WorkQueue,
)

from ..attn_block_cache import AttnBlockCacheEntry, AttnBlockCache
from ..config import ServiceParams
from ..service import (
    BatchGenerateService,
    BatchGenerateState,
    GenerateRequest,
)


logger = get_logger("shortfin.llm.impl.service_v1")

EXPECTED_CONCURRENCY = 10


class GenerateServiceV1(BatchGenerateService):
    def __init__(
        self, *, session: DeviceSession, params: ServiceParams, cache: AttnBlockCache
    ):
        self.params = params
        self.block_pos_stride = params.cache.block_pos_stride
        self.batch_sizes = params.model.prefill_batch_sizes
        # TODO: Remove distinction between prefill and decode batch sizes.
        assert params.model.decode_batch_sizes == self.batch_sizes
        self.session = session
        self.cache = cache
        module_name = params.model.module_name
        logger.info("Configuring serving for module set %s", module_name)
        self.module_set = session.module_set(params.model.module_name)

        # Initialize prefill entry-points (1 per batch size).
        self.prefill_functions: dict[int, VmFunction] = {}
        for bs in self.batch_sizes:
            assert bs not in self.prefill_functions
            symbol_name = f"prefill_bs{bs}"
            logger.info("Looking up symbol '%s'", symbol_name)
            self.prefill_functions[bs] = self.module_set.function(
                module_name, symbol_name
            )

        # Initialize decode entry-points (1 per batch size).
        self.decode_functions: dict[int, VmFunction] = {}
        for bs in self.batch_sizes:
            assert bs not in self.decode_functions
            symbol_name = f"decode_bs{bs}"
            logger.info("Looking up symbol '%s'", symbol_name)
            self.decode_functions[bs] = self.module_set.function(
                module_name, symbol_name
            )

        self._initialize_transfer_pools()

    def _initialize_transfer_pools(self):
        params = self.params
        max_bs = params.model.max_batch_size
        max_sl = params.model.max_seq_len
        initial_inflight = EXPECTED_CONCURRENCY

        # block_indices_pool: array([max_batch_size, max_attn_blocks], np.int64)
        # Suitable to handle the sequence->block mapping for all steps.
        self.block_indices_pool = TransferBufferPool.shaped(
            self.session,
            [
                max_bs,
                max_sl // self.block_pos_stride,
            ],
            HalElementType.SINT_64,
            initial_capacity=initial_inflight,
            growable=True,
            name="block_cache_indices",
        )

        # Prefill tokens: array([max_batch_size, max_seq_len], np.int64)
        # Tokens inputs to prefill.
        self.prefill_tokens_pool = TransferBufferPool.shaped(
            self.session,
            [
                max_bs,
                max_sl,
            ],
            HalElementType.SINT_64,
            initial_capacity=initial_inflight,
            growable=True,
            name="prefill_tokens",
        )

        # Prefill sequence lengths: array([max_batch_size], np.int64)
        # Sequence lengths of input tokens.
        self.prefill_seq_lens_pool = TransferBufferPool.shaped(
            self.session,
            [max_bs],
            HalElementType.SINT_64,
            initial_capacity=initial_inflight,
            growable=True,
            name="prefill_seq_lens",
        )

        # Decode tokens: array([max_batch_size], np.int64)
        # Tokens to perform a decode step with.
        self.decode_tokens_pool = TransferBufferPool.shaped(
            self.session,
            [max_bs, 1],
            HalElementType.SINT_64,
            initial_capacity=initial_inflight,
            growable=True,
            name="decode_tokens",
        )

        # Decode seq lengths: array([max_batch_size], np.int64)
        # Decoder seq length for this step
        self.decode_seq_lens_pool = TransferBufferPool.shaped(
            self.session,
            [max_bs],
            HalElementType.SINT_64,
            initial_capacity=initial_inflight,
            growable=True,
            name="decode_seq_len",
        )

        # Decode start positions: array([max_batch_size], np.int64)
        # Tokens to perform a decode step with.
        self.decode_start_pos_pool = TransferBufferPool.shaped(
            self.session,
            [max_bs],
            HalElementType.SINT_64,
            initial_capacity=initial_inflight,
            growable=True,
            name="decode_start_pos",
        )

    def start(self) -> "GenerateState":
        return GenerateState(self)

    def shutdown(self):
        self.session.shutdown()


class _Sequence:
    __slots__ = [
        "attn_blocks",
        "attn_blocks_needed",
        "current_token_ids",
        "decode_token_ids",
        "request",
        "seq_length",
    ]

    current_token_ids: list[int]
    decode_token_ids: list[int]

    def __init__(self, request: GenerateRequest):
        self.request = request
        self.seq_length: int = 0
        self.attn_blocks: list[AttnBlockCacheEntry] = []
        self.attn_blocks_needed: int = 0
        self.decode_token_ids = []
        self.current_token_ids = []

    def attn_blocks_available(self):
        return len(self.attn_blocks)

    def resize_attention(self, new_size):
        old_size = self.attn_blocks_needed
        self.attn_blocks_needed = new_size
        return new_size - old_size


class GenerateState(BatchGenerateState):
    __slots__ = [
        "_bs",
        "_decode_function",
        "_prefill_function",
        "_max_attn_blocks_length",
        "_max_seq_length",
        "_resources",
        "_service",
        "_sequences",
        "_batch_queue",
    ]

    def __init__(self, service: GenerateServiceV1):
        super().__init__(service.module_set.host_context)
        self._resources = AsyncResources()
        self._service = service
        self._sequences: list[_Sequence] = []
        self._batch_queue = WorkQueue(service.session)

    async def recycle(self):
        """Recycles or releases all resources consumed by this instance."""
        cache = self._service.cache
        self._batch_queue.sync(self.host_context)
        self._resources.recycle()
        all_blocks = []
        for seq in self._sequences:
            all_blocks.extend(seq.attn_blocks)
            seq.attn_blocks.clear()
        self._sequences = []
        await cache.release_attn_blocks(all_blocks)

    async def set_sequences(self, requests: list[GenerateRequest]):
        """Initiates processing of a list of sequences that make up a batch.

        This is async because it acquires resources which may not be available.
        """
        service = self._service
        block_pos_stride = service.block_pos_stride

        # Loop through each request and reserve initial attention blocks.
        bs = 0
        sequences = self._sequences
        assert not sequences, "set_sequences already called"
        max_attn_blocks_length = 0
        max_seq_length = 0
        attn_blocks_required = 0

        for req in requests:
            bs += 1
            seq = _Sequence(req)
            sequences.append(seq)
            seq.current_token_ids = req.required_prompt_token_ids
            seq_length = len(seq.current_token_ids)
            seq.seq_length = seq_length
            max_seq_length = max(max_seq_length, seq_length)
            initial_block_count = seq_length // block_pos_stride + 1
            attn_blocks_required += initial_block_count
            seq.attn_blocks_needed = initial_block_count
            max_attn_blocks_length = max(max_attn_blocks_length, initial_block_count)

        # Determine the appropriate batched entrypoints.
        assert bs > 0
        for allowed_bs in service.batch_sizes:
            if allowed_bs >= bs:
                self._prefill_function = service.prefill_functions[allowed_bs]
                self._decode_function = service.decode_functions[allowed_bs]
                break
        else:
            raise AssertionError(f"Unsupported batch size: {bs}")

        # Acquire the needed attention blocks in one batch so as to give the scheduler
        # the most visibility into the need.
        logger.debug("Acquire prefill attn blocks: %s", attn_blocks_required)
        all_attn_blocks: list[AttnBlockCacheEntry] = []
        await service.cache.acquire_attn_blocks(attn_blocks_required, all_attn_blocks)
        block_index = 0
        for seq in sequences:
            next_block_count = seq.attn_blocks_needed
            seq.attn_blocks.extend(
                all_attn_blocks[block_index : block_index + seq.attn_blocks_needed]
            )
            block_index += next_block_count

        # Save state.
        self._bs = allowed_bs
        self._max_attn_blocks_length = max_attn_blocks_length
        self._max_seq_length = max_seq_length

    async def prefill(self) -> TimelineGuarded[HalBufferView]:
        hc = self.host_context
        service = self._service
        resources = self._resources
        bs = self._bs
        service = self._service
        block_pos_stride = service.block_pos_stride
        max_attn_blocks_length = self._max_attn_blocks_length
        max_seq_length = max_attn_blocks_length * block_pos_stride
        sequences = self._sequences
        work_queue = self._batch_queue

        # Record a command buffer for performing h2d transfers.
        cb = HalCommandBuffer(hc.session.device)

        # Prepare input tokens, sequence lengths and block indices.
        # We acquire a transfer buffer of each from the respective pool, populate its
        # host side and enqueue.
        # prefill_tokens: array([bs, max_seq_length], np.int32)
        prefill_tokens_host, prefill_tokens_device = resources.acquire_transfer_buffer(
            service.prefill_tokens_pool
        ).h2d_array(cb, [bs, max_seq_length], HalElementType.SINT_64, fill_value=0)

        # prefill_seq_lens: array([bs], np.int32)
        (
            prefill_seq_lens_host,
            prefill_seq_lens_device,
        ) = resources.acquire_transfer_buffer(service.prefill_seq_lens_pool).h2d_array(
            cb, [bs], HalElementType.SINT_64, fill_value=0
        )

        # attn_block_indices: array([bs, max_attn_blocks], np.in16)
        (
            prefill_attn_block_indices_host,
            prefill_attn_block_indices_device,
        ) = resources.acquire_transfer_buffer(service.block_indices_pool).h2d_array(
            cb, [bs, max_attn_blocks_length], HalElementType.SINT_64, fill_value=0
        )

        # Populate host buffers for each sequence.
        for i in range(len(sequences)):
            seq = sequences[i]
            attn_blocks = seq.attn_blocks
            current_token_ids = seq.current_token_ids
            row_seq_len = len(current_token_ids)
            prefill_tokens_host[i, 0:row_seq_len] = current_token_ids
            prefill_seq_lens_host[i] = row_seq_len
            for j in range(len(seq.attn_blocks)):
                prefill_attn_block_indices_host[i, j] = attn_blocks[j].index

        # Perform h2d transfers.
        cb.end()
        work_queue.execute_sequential([cb])

        # Inputs:
        #   token_ids
        #   seq_lens
        #   attn_block_indices
        #   attn_block_buffer_view (the entire slab passed as input)
        #   wait, signal semaphores
        #   tied attn_block_buffer (for input[2])
        #   tied attn_block_buffer (for result[0])
        inputs = VmVariantList(3)
        inputs.push_ref(prefill_tokens_device)
        inputs.push_ref(prefill_seq_lens_device)
        inputs.push_ref(prefill_attn_block_indices_device)
        inputs.push_ref(service.cache.attn_block_buffer_view)

        # Outputs:
        #   attn_block_buffer_view (tied output)
        #   decode_tokens
        outputs = VmVariantList(1)
        # TODO: Async invoke.
        hc.vm_context.invoke(self._prefill_function, inputs, outputs)
        return work_queue.guard(outputs.get_as_ref(0).deref(HalBufferView))

    async def set_decode_step(self, tokens):
        """Initiates processing of a list of tokens to decode across each batch

        This is async because it acquires resources which may not be available.
        """
        service = self._service
        block_pos_stride = service.block_pos_stride

        sequences = self._sequences
        assert sequences, "set_sequences was not called yet"
        assert len(sequences) == len(tokens), "expected token for each sequence"

        max_attn_blocks_length = 0
        max_seq_length = 0
        attn_blocks_required = 0

        for tok, seq in zip(tokens, self._sequences):
            seq.decode_token_ids.append(tok)
            seq.seq_length = seq.seq_length + 1

            max_seq_length = max(max_seq_length, seq.seq_length)
            block_count = seq.seq_length // block_pos_stride + 1

            seq.attn_blocks_needed = block_count
            attn_blocks_required += block_count - seq.attn_blocks_available()
            max_attn_blocks_length = max(max_attn_blocks_length, block_count)

        # Acquire the needed attention blocks in one batch so as to give the scheduler
        # the most visibility into the need.
        logger.debug("Acquire decode attn blocks: %s", attn_blocks_required)
        all_attn_blocks: list[AttnBlockCacheEntry] = []
        await service.cache.acquire_attn_blocks(attn_blocks_required, all_attn_blocks)
        block_index = 0
        for seq in sequences:
            next_block_count = seq.attn_blocks_needed - seq.attn_blocks_available()
            seq.attn_blocks.extend(
                all_attn_blocks[block_index : block_index + next_block_count]
            )
            block_index += next_block_count

        # Save state.
        self._max_attn_blocks_length = max_attn_blocks_length
        self._max_seq_length = max_seq_length

    async def decode(self) -> TimelineGuarded[HalBufferView]:
        hc = self.host_context
        service = self._service
        resources = self._resources
        bs = self._bs
        max_attn_blocks_length = self._max_attn_blocks_length
        sequences = self._sequences
        work_queue = self._batch_queue

        # Record a command buffer for performing h2d transfers.
        cb = HalCommandBuffer(hc.session.device)

        # decode_tokens: array([bs, 1], np.int32)
        (decode_tokens_host, decode_tokens_device,) = resources.acquire_transfer_buffer(
            service.decode_tokens_pool
        ).h2d_array(cb, [bs, 1], HalElementType.SINT_64, fill_value=0)

        # decode_seq_lens: array([bs], np.int32)
        (
            decode_seq_lens_host,
            decode_seq_lens_device,
        ) = resources.acquire_transfer_buffer(service.decode_seq_lens_pool).h2d_array(
            cb, [bs], HalElementType.SINT_64, fill_value=0
        )

        # decode_start_pos: array([bs], np.int32)
        (
            decode_start_pos_host,
            decode_start_pos_device,
        ) = resources.acquire_transfer_buffer(service.decode_start_pos_pool).h2d_array(
            cb, [bs], HalElementType.SINT_64, fill_value=0
        )

        # attn_block_indices: array([bs, max_attn_blocks], np.in16)
        (
            decode_attn_block_indices_host,
            decode_attn_block_indices_device,
        ) = resources.acquire_transfer_buffer(service.block_indices_pool).h2d_array(
            cb, [bs, max_attn_blocks_length], HalElementType.SINT_64, fill_value=0
        )

        # Populate host buffers for each sequence.
        for i in range(len(sequences)):
            seq = sequences[i]
            attn_blocks = seq.attn_blocks

            tok = seq.decode_token_ids[0]
            seq_len = len(seq.current_token_ids)
            print(seq.current_token_ids)
            seq.current_token_ids.append(tok)
            seq.decode_token_ids = seq.decode_token_ids[1:]

            decode_tokens_host[i, 0] = tok
            decode_start_pos_host[i] = seq_len
            decode_seq_lens_host[i] = seq_len
            for j in range(len(seq.attn_blocks)):
                decode_attn_block_indices_host[i, j] = attn_blocks[j].index

        # Perform h2d transfers.
        cb.end()
        work_queue.execute_sequential([cb])

        # Inputs:
        #   token_ids
        #   seq_lens
        #   start_pos
        #   attn_block_indices
        #   attn_block_buffer_view (the entire slab passed as input)
        #   wait, signal semaphores
        #   tied attn_block_buffer (for input[4])
        #   tied attn_block_buffer (for result[0])
        inputs = VmVariantList(5)
        inputs.push_ref(decode_tokens_device)
        inputs.push_ref(decode_seq_lens_device)
        inputs.push_ref(decode_start_pos_device)
        inputs.push_ref(decode_attn_block_indices_device)
        inputs.push_ref(service.cache.attn_block_buffer_view)

        # Outputs:
        #   attn_block_buffer_view (tied output)
        #   decode_tokens
        outputs = VmVariantList(1)
        # TODO: Async invoke.
        hc.vm_context.invoke(self._decode_function, inputs, outputs)
        return work_queue.guard(outputs.get_as_ref(0).deref(HalBufferView))
