# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import logging
from pathlib import Path

import shortfin as sf
import shortfin.array as sfnp

# TODO: Have a generic "Responder" interface vs just the concrete impl.
from shortfin.interop.fastapi import FastAPIResponder

from .cache import AttnPageCache
from .config_struct import ModelParams
from .manager import SystemManager
from .messages import PrefillRequest, StrobeMessage
from .tokenizer import Encoding, Tokenizer

logger = logging.getLogger(__name__)


class GenerateService:

    inference_program: sf.Program
    prefill_functions: dict[int, sf.ProgramFunction]

    def __init__(
        self,
        *,
        name: str,
        sysman: SystemManager,
        tokenizer: Tokenizer,
        model_params: ModelParams,
    ):
        self.name = name

        # Application objects.
        self.sysman = sysman
        self.tokenizer = tokenizer
        self.model_params = model_params
        self.inference_parameters: list[sf.BaseProgramParameters] = []
        self.inference_modules: list[sf.ProgramModule] = []

        self.main_worker = sysman.ls.create_worker(f"{name}-inference")
        self.main_scope = sysman.ls.create_scope(self.main_worker)

        # Scope dependent objects.
        self.batcher = BatcherProcess(self)
        self.page_cache = AttnPageCache(
            devices=self.main_scope.devices_dict.values(), model_params=model_params
        )

    def load_inference_module(self, vmfb_path: Path):
        self.inference_modules.append(sf.ProgramModule.load(self.sysman.ls, vmfb_path))

    def load_inference_parameters(
        self, *paths: Path, parameter_scope: str, format: str = ""
    ):
        p = sf.StaticProgramParameters(self.sysman.ls, parameter_scope=parameter_scope)
        for path in paths:
            logging.info("Loading parameter scope '%s' from: %s", parameter_scope, path)
            p.load(path, format=format)
        self.inference_parameters.append(p)

    def start(self):
        self.inference_program = sf.Program(
            modules=[
                sf.ProgramModule.parameter_provider(
                    self.sysman.ls, *self.inference_parameters
                )
            ]
            + self.inference_modules,
            scope=self.main_scope,
            trace_execution=False,
        )
        # Resolve prefill entrypoints.
        self.prefill_functions = {}
        for bs in self.model_params.prefill_batch_sizes:
            self.prefill_functions[bs] = self.inference_program[
                f"{self.model_params.module_name}.prefill_bs{bs}"
            ]

        self.batcher.launch()

    def shutdown(self):
        self.batcher.shutdown()

    def __repr__(self):
        return (
            f"ServiceManager(\n"
            f"  model_params={self.model_params}\n"
            f"  inference_modules={self.inference_modules}\n"
            f"  page_cache={self.page_cache}\n"
            f")"
        )


########################################################################################
# Batcher
########################################################################################

import math


class BatcherProcess(sf.Process):
    """The batcher is a persistent process responsible for flighting incoming work
    into batches and handling the requisite cache allocations (since every batch needs
    committed cache state).
    """

    STROBE_SHORT_DELAY = 0.1
    STROBE_LONG_DELAY = 0.25

    def __init__(self, service: GenerateService):
        super().__init__(scope=service.main_scope)
        self.service = service
        self.batcher_infeed = self.system.create_queue()
        self.pending_prefills: set[PrefillRequest] = set()
        self.strobe_enabled = True
        self.strobes: int = 0
        self.ideal_batch_size: int = max(service.model_params.prefill_batch_sizes)
        self.page_seq_stride = service.model_params.paged_kv_cache.block_seq_stride

    def shutdown(self):
        self.batcher_infeed.close()

    def submit(self, request: StrobeMessage | PrefillRequest):
        self.batcher_infeed.write_nodelay(request)

    async def _background_strober(self):
        while not self.batcher_infeed.closed:
            await asyncio.sleep(
                BatcherProcess.STROBE_SHORT_DELAY
                if len(self.pending_prefills) > 0
                else BatcherProcess.STROBE_LONG_DELAY
            )
            if self.strobe_enabled:
                self.submit(StrobeMessage())

    async def run(self):
        strober_task = asyncio.create_task(self._background_strober())
        reader = self.batcher_infeed.reader()
        while item := await reader():
            self.strobe_enabled = False
            if isinstance(item, PrefillRequest):
                self.pending_prefills.add(item)
            elif isinstance(item, StrobeMessage):
                self.strobes += 1
            else:
                logger.error("Illegal message received by batcher: %r", item)
            self.board_flights()
            self.strobe_enabled = True
        await strober_task

    def board_flights(self):
        waiting_count = len(self.pending_prefills)
        if waiting_count == 0:
            return
        if waiting_count < self.ideal_batch_size and self.strobes < 2:
            logger.info("Waiting a bit longer to fill flight")
            return
        self.strobes = 0

        # Fill prefill flights.
        # TODO: This is a very naive cache management algorithm. Burn with fire
        # and implement a real one.
        cache = self.service.page_cache
        prefill_process = PrefillExecutorProcess(
            self.service, self.page_seq_stride, cache.page_tables
        )
        logger.info("Flight taking off with %d souls", waiting_count)
        for prefill_request in self.pending_prefills:
            if len(prefill_process.prefill_requests) >= self.ideal_batch_size:
                break
            needed_pages = math.ceil(
                len(prefill_request.input_token_ids) / self.page_seq_stride
            )
            pages = cache.acquire_free_pages(needed_pages)
            if pages is None:
                logger.debug("Cannot fulfill request for %d pages", needed_pages)
                continue
            else:
                logger.debug("Allocated %d cache pages to request", len(pages))
                prefill_request.lock_cache_pages(cache, pages)

            # Can flight this request.
            prefill_process.prefill_requests.append(prefill_request)

        # We've filled our flight. Remove from the boarding area.
        if prefill_process.prefill_requests:
            for flighted_request in prefill_process.prefill_requests:
                self.pending_prefills.remove(flighted_request)
            # And takeoff.
            prefill_process.launch()

        # For now, kill anything that is left.
        for prefill_request in self.pending_prefills:
            prefill_request.done.set_success()
        self.pending_prefills.clear()
        logger.debug("Post boarding cache state: %r", cache)


########################################################################################
# Prefill Executor
########################################################################################


class PrefillExecutorProcess(sf.Process):
    """Executes a prefill batch."""

    def __init__(self, service: GenerateService, seq_stride: int, page_tables):
        super().__init__(scope=service.main_scope)
        self.service = service
        self.seq_stride = seq_stride
        self.prefill_requests: list[PrefillRequest] = []
        self.page_tables = page_tables

    async def run(self):
        try:
            req_bs = len(self.prefill_requests)
            seq_stride = self.seq_stride
            # Select an entrypoint for the batch.
            for bs, fn in self.service.prefill_functions.items():
                if bs >= req_bs:
                    break
            else:
                raise RuntimeError(f"No available entry point for bs {req_bs}")

            # Compute block sequence length as maximum sequence length, rounded
            # up to the seq_stride.
            bsl = max(len(r.input_token_ids) for r in self.prefill_requests)
            bsl = int(math.ceil(bsl / seq_stride) * seq_stride)
            block_count = bsl // seq_stride
            req_count = len(self.prefill_requests)
            logger.debug("Prefill bs=%d, bsl=%d", bs, bsl)

            # Prepare inputs.
            # TODO: Better support in shortfin for h2d. The best way to do it is
            # device dependent.
            device0 = self.scope.device(0)
            int_dtype = sfnp.int64
            tokens = sfnp.device_array.for_device(device0, [bs, bsl], int_dtype)
            seq_lens = sfnp.device_array.for_device(device0, [bs], int_dtype)
            seq_block_ids = sfnp.device_array.for_device(
                device0, [bs, block_count], int_dtype
            )

            # Populate tokens.
            tokens_host = tokens.for_transfer()
            for i in range(bs):
                with tokens_host.view(i).map(discard=True) as m:
                    m.fill(0)
                    if i < req_count:
                        m.items = self.prefill_requests[i].input_token_ids
            tokens_host.copy_to(tokens)

            # Populate seq_lens.
            seq_lens_host = seq_lens.for_transfer()
            with seq_lens_host.map(discard=True) as m:
                m.fill(0)
                m.items = [len(req.input_token_ids) for req in self.prefill_requests]
            seq_lens_host.copy_to(seq_lens)

            # Populate cache pages.
            seq_block_ids_host = seq_block_ids.for_transfer()
            for i in range(bs):
                with seq_block_ids_host.view(i).map(discard=True) as m:
                    m.fill(0)
                    if i < req_count:
                        m.items = self.prefill_requests[i].cache_page_indices(
                            block_count
                        )
            seq_block_ids_host.copy_to(seq_block_ids)

            # Invoke. Logits are of shape [bs, bsl, d].
            (logits,) = await fn(tokens, seq_lens, seq_block_ids, *self.page_tables)

            # Return results.
            for i in range(req_count):
                req = self.prefill_requests[i]
                sl = len(req.input_token_ids)
                if req.return_all_logits:
                    logits_item = logits.view(i, slice(0, sl))
                else:
                    logits_item = logits.view(i, sl - 1)
                if req.return_host_array:
                    req.result_logits = logits_item.for_transfer()
                    req.result_logits.copy_from(logits_item)
                    await device0
                else:
                    req.result_logits = logits_item
                req.done.set_success()

        except Exception as e:
            logger.exception("Fatal error in prefetch invocation")
            # TODO: Cancel and set error correctly
            for req in self.prefill_requests:
                req.result_logits = None
                req.free_cache_pages()
                req.done.set_success()
