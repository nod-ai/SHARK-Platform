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

from .config_struct import ModelParams
from .manager import SystemManager
from .messages import InferenceExecRequest, InferencePhase, StrobeMessage
from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class GenerateService:
    """Top level service interface for image generation."""

    encode_program: sf.Program
    denoise_program: sf.Program
    decode_program: sf.Program

    encode_functions: dict[int, sf.ProgramFunction]
    denoise_functions: dict[int, sf.ProgramFunction]
    decode_functions: dict[int, sf.ProgramFunction]

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
        self.main_fiber = sysman.ls.create_fiber(self.main_worker)

        # Scope dependent objects.
        self.batcher = BatcherProcess(self)

    def load_inference_module(self, vmfb_path: Path):
        self.inference_modules.append(sf.ProgramModule.load(self.sysman.ls, vmfb_path))

    def load_inference_parameters(
        self, *paths: Path, parameter_scope: str, format: str = ""
    ):
        p = sf.StaticProgramParameters(self.sysman.ls, parameter_scope=parameter_scope)
        for path in paths:
            logging.info("Loading parameter fiber '%s' from: %s", parameter_scope, path)
            p.load(path, format=format)
        self.inference_parameters.append(p)

    def start(self):
        # Initiate inference programs.
        # Resolve all function entrypoints.
        # Start persistent processes.
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
        super().__init__(fiber=service.main_fiber)
        self.service = service
        self.batcher_infeed = self.system.create_queue()
        self.pending_preps: set[InferenceExecRequest] = set()
        self.pending_encodes: set[InferenceExecRequest] = set()
        self.pending_denoises: set[InferenceExecRequest] = set()
        self.pending_decodes: set[InferenceExecRequest] = set()
        self.pending_postprocesses: set[InferenceExecRequest] = set()
        self.strobe_enabled = True
        self.strobes: int = 0
        self.ideal_batch_size: int = max(service.model_params.submodel_batch_sizes)

    def shutdown(self):
        self.batcher_infeed.close()

    def submit(self, request: StrobeMessage | InferenceExecRequest):
        self.batcher_infeed.write_nodelay(request)

    async def _background_strober(self):
        while not self.batcher_infeed.closed:
            await asyncio.sleep(
                BatcherProcess.STROBE_SHORT_DELAY
                if len(self.pending_preps) > 0
                else BatcherProcess.STROBE_LONG_DELAY
            )
            if self.strobe_enabled:
                self.submit(StrobeMessage())

    async def run(self):
        strober_task = asyncio.create_task(self._background_strober())
        reader = self.batcher_infeed.reader()
        while item := await reader():
            self.strobe_enabled = False
            if isinstance(item, InferenceExecRequest):
                phase = item.phase
                if phase == InferencePhase.PREP:
                    self.pending_preps.add(item)
                elif phase == InferencePhase.ENCODE:
                    self.pending_encodes.add(item)
                elif phase == InferencePhase.DENOISE:
                    self.pending_denoises.add(item)
                elif phase == InferencePhase.DECODE:
                    self.pending_decodes.add(item)
                elif phase == InferencePhase.POSTPROCESS:
                    self.pending_postprocesses.add(item)
                else:
                    logger.error("Illegal InferenceExecRequest phase: %r", phase)
            elif isinstance(item, StrobeMessage):
                self.strobes += 1
            else:
                logger.error("Illegal message received by batcher: %r", item)
            self.board_flights()
            self.strobe_enabled = True
        await strober_task

    def board_flights(self):
        waiting_count = (
            len(self.pending_preps) 
            + len(self.pending_encodes)
            + len(self.pending_denoises)
            + len(self.pending_decodes)
            + len(self.pending_postprocesses)
        )
        if waiting_count == 0:
            return
        if waiting_count < self.ideal_batch_size and self.strobes < 2:
            logger.info("Waiting a bit longer to fill flight")
            return
        self.strobes = 0

        self.board_preps()
        self.board_encodes()
        self.board_denoises()
        self.board_decodes()
        self.board_postprocesses()

        # For now, kill anything that is left.
        for i in [
            self.pending_preps,
            self.pending_encodes,
            self.pending_denoises,
            self.pending_decodes,
            self.pending_postprocesses,
        ]:
            for request in i:
                request.done.set_success()
            i.clear()

    def board_preps():
        pass
    
    def board_encodes():
        pass
    
    def board_denoises():
        pass
    
    def board_decodes():
        pass
    
    def board_postprocesses():
        pass


########################################################################################
# Inference Executors
########################################################################################


class InferenceExecProcessPrep(sf.Process):
    """Executes a prep batch"""

    def __init__(
        self,
        service: GenerateService,
    ):
        super().__init__(fiber=service.main_fiber)
        self.service = service
        self.exec_requests: list[InferenceExecRequest] = []

    async def run(self):
        try:
            for i in range(req_count):
                req.done.set_success()

        except Exception:
            logger.exception("Fatal error in inputs preparation")
            # TODO: Cancel and set error correctly
            for req in self.exec_requests:
                req.payload = None
                req.done.set_success()

class InferenceExecProcessEncode(sf.Process):
    """Executes a CLIP encoding batch"""

    def __init__(
        self,
        service: GenerateService,
    ):
        super().__init__(fiber=service.main_fiber)
        self.service = service
        self.exec_requests: list[InferenceExecRequest] = []

    async def run(self):
        try:
            for i in range(req_count):
                req.done.set_success()

        except Exception:
            logger.exception("Fatal error in text encoding invocation")
            # TODO: Cancel and set error correctly
            for req in self.exec_requests:
                req.payload = None
                req.done.set_success()

class InferenceExecProcessDenoise(sf.Process):
    """Executes a denoising loop for a batch of latents."""

    def __init__(
        self,
        service: GenerateService,
    ):
        super().__init__(fiber=service.main_fiber)
        self.service = service
        self.exec_requests: list[InferenceExecRequest] = []

    async def run(self):
        try:
            for i in range(req_count):
                req.done.set_success()

        except Exception:
            logger.exception("Fatal error in denoising loop invocation.")
            # TODO: Cancel and set error correctly
            for req in self.exec_requests:
                req.payload = None
                req.done.set_success()

class InferenceExecProcessDecode(sf.Process):
    """Executes decoding for a batch of denoised latents."""

    def __init__(
        self,
        service: GenerateService,
    ):
        super().__init__(fiber=service.main_fiber)
        self.service = service
        self.exec_requests: list[InferenceExecRequest] = []

    async def run(self):
        try:
            for i in range(req_count):
                req.done.set_success()

        except Exception:
            logger.exception("Fatal error in decoding image latents.")
            # TODO: Cancel and set error correctly
            for req in self.exec_requests:
                req.payload = None
                req.done.set_success()

class InferenceExecProcessPostprocess(sf.Process):
    """Executes postprocessing for a batch of generated images."""

    def __init__(
        self,
        service: GenerateService,
    ):
        super().__init__(fiber=service.main_fiber)
        self.service = service
        self.exec_requests: list[InferenceExecRequest] = []

    async def run(self):
        try:
            for i in range(req_count):
                req.done.set_success()

        except Exception:
            logger.exception("Fatal error in postprocessing images.")
            # TODO: Cancel and set error correctly
            for req in self.exec_requests:
                req.payload = None
                req.done.set_success()


