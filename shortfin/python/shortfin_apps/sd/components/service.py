# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import logging
import math
from pathlib import Path
from PIL import Image

import torch

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
        tokenizers: list[Tokenizer],
        model_params: ModelParams,
    ):
        self.name = name

        # Application objects.
        self.sysman = sysman
        self.tokenizers = tokenizers
        self.model_params = model_params
        self.inference_parameters: list[sf.BaseProgramParameters] = []
        self.inference_modules: list[sf.ProgramModule] = []

        self.encode_functions: dict[int, sf.ProgramFunction] = {}
        self.denoise_functions: dict[int, sf.ProgramFunction] = {}
        self.decode_functions: dict[int, sf.ProgramFunction] = {}

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
            f")"
        )


########################################################################################
# Batcher
########################################################################################


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
        self.phase_map = {
            InferencePhase.PREPARE: self.pending_preps,
            InferencePhase.ENCODE: self.pending_encodes,
            InferencePhase.DENOISE: self.pending_denoises,
            InferencePhase.DECODE: self.pending_decodes,
            InferencePhase.POSTPROCESS: self.pending_postprocesses,
        }
        self.strobe_enabled = True
        self.strobes: int = 0
        self.ideal_batch_size: int = max(service.model_params.max_batch_size)

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
                if phase in self.phase_map.keys():
                    self.phase_map[phase].add(item)
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
        waiting_count = sum([len(val) for val in self.phase_map.values()])
        if waiting_count == 0:
            return
        if waiting_count < self.ideal_batch_size and self.strobes < 2:
            logger.info("Waiting a bit longer to fill flight")
            return
        self.strobes = 0

        for phase in self.phase_map.keys():
            self.board(phase)

        # For now, kill anything that is left.
        for i in self.phase_map.values():
            for request in i:
                request.done.set_success()
            i.clear()

    def board(self, phase: InferencePhase):
        pending = self.phase_map[phase]
        if len(pending) == 0:
            return
        exec_process = InferenceExecutorProcess(self.service)
        for req in pending:
            assert req.phase == phase
            if len(exec_process.exec_requests) >= self.ideal_batch_size:
                break
            exec_process.exec_requests.append(req)
        if exec_process.exec_requests:
            for flighted_request in exec_process.exec_requests:
                self.phase_map[phase].remove(flighted_request)
            exec_process.launch()


########################################################################################
# Inference Executors
########################################################################################


class InferenceExecutorProcess(sf.Process):
    """Executes a stable diffusion inference batch"""

    def __init__(
        self,
        service: GenerateService,
    ):
        super().__init__(fiber=service.main_fiber)
        self.service = service
        self.exec_requests: list[InferenceExecRequest] = []

    async def run(self):
        try:
            phase = None
            for req in self.exec_requests:
                if phase:
                    assert phase == req.phase
                phase = req.phase
            is_prep = phase == InferencePhase.PREPARE
            is_encode = phase == InferencePhase.ENCODE
            is_denoise = phase == InferencePhase.DENOISE
            is_decode = phase == InferencePhase.DECODE
            is_postprocess = phase == InferencePhase.POSTPROCESS
            req_count = len(self.exec_requests)
            device0 = self.fiber.device(0)

            if is_prep:
                await self._prepare(device=device0, requests=self.exec_requests)
            elif is_encode:
                await self._encode(device=device0, requests=self.exec_requests)
            elif is_denoise:
                await self._denoise(device=device0, requests=self.exec_requests)
            elif is_decode:
                await self._decode(device=device0, requests=self.exec_requests)
            elif is_postprocess:
                await self._postprocess(device=device0, requests=self.exec_requests)

            for i in range(req_count):
                req = self.exec_requests[i]
                req.done.set_success()

        except Exception:
            logger.exception("Fatal error in image generation")
            # TODO: Cancel and set error correctly
            for req in self.exec_requests:
                req.done.set_success()

    async def _prepare(self, device, requests):
        torch_dtypes = {
            sfnp.float16: torch.float16,
            sfnp.float32: torch.float32,
            sfnp.int8: torch.int8,
            sfnp.bfloat16: torch.bfloat16,
        }
        for request in requests:
            # Tokenize prompts and negative prompts. We tokenize in bs1 for now and join later.
            input_ids_list = []
            prompts = [
                request.prompt,
                request.neg_prompt,
            ]
            for tokenizer, prompt in zip(self.service.tokenizers, prompts):
                input_ids = tokenizer.encode(prompt)
                input_ids_list.append(input_ids)

            request.input_ids = input_ids_list

            # Generate random sample latents.
            # TODO: use our own RNG
            seed = request.seed
            channels = self.service.model_params.num_latents_channels
            unet_dtype = self.service.model_params.unet_dtype
            latents_shape = (
                1,
                channels,
                request.height // 8,
                request.width // 8,
            )
            generator = torch.manual_seed(seed)
            rand_sample = torch.randn(
                latents_shape,
                generator=generator,
                dtype=torch_dtypes[unet_dtype],
            ).numpy()

            # Create and populate sample device array.
            request.sample = sfnp.device_array.for_device(
                device, latents_shape, unet_dtype
            )
            sample_host = request.sample.for_transfer()
            with sample_host.map(discard=True) as m:
                m.fill(rand_sample)

            # Guidance scale.
            guidance_scale = sfnp.device_array.for_device(device, [1], unet_dtype)
            guidance_scale_float = sfnp.device_array.for_device(
                device, [1], sfnp.float32
            )
            guidance_float_host = guidance_scale_float.for_transfer()
            with guidance_float_host.map(discard=True) as m:
                m.items = [request.guidance_scale]
            guidance_host = guidance_scale.for_transfer()
            with guidance_host.map(discard=True) as m:
                guidance_scale_float.copy_to(guidance_host)
            request.guidance_scale = guidance_scale
        return

    async def _encode(self, device, requests):
        req_bs = len(requests)
        # Encode inputs
        entrypoints = self.service.encode_functions
        for bs, fn in entrypoints.items():
            if bs >= req_bs:
                break
        return

    async def _denoise(self, device, requests):
        req_bs = len(requests)
        # Produce denoised latents
        entrypoints = self.service.denoise_functions
        for bs, fn in entrypoints.items():
            if bs >= req_bs:
                break
        return

    async def _decode(self, device, requests):
        req_bs = len(requests)
        # Decode latents to images
        entrypoints = self.service.decode_functions
        for bs, fn in entrypoints.items():
            if bs >= req_bs:
                func = fn
                break
        return

    async def _postprocess(self, device, requests):
        # Process output images
        for req in requests:
            req.result_image = Image.open("sample.png", mode="r").tobytes()
        return
