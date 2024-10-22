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

    inference_programs: dict[str, sf.Program]

    inference_functions: dict[str, dict[str, sf.ProgramFunction]]

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
        self.inference_parameters: dict[str, list[sf.BaseProgramParameters]] = {}
        self.inference_modules: dict[str, sf.ProgramModule] = {}
        self.inference_functions: dict[str, dict[str, sf.ProgramFunction]] = {}
        self.inference_programs: dict[str, sf.Program] = {}
        self.workers = []
        self.fibers = []
        for idx, device in enumerate(self.sysman.ls.devices):
            worker = sysman.ls.create_worker(f"{name}-inference-{device.name}")
            fiber = sysman.ls.create_fiber(worker, devices=[device])
            self.workers.append(worker)
            self.fibers.append(fiber)

        # Scope dependent objects.
        self.batcher = BatcherProcess(self)

    def load_inference_module(self, vmfb_path: Path, component: str = None):
        if not self.inference_modules.get(component):
            self.inference_modules[component] = []
        self.inference_modules[component].append(
            sf.ProgramModule.load(self.sysman.ls, vmfb_path)
        )

    def load_inference_parameters(
        self,
        *paths: Path,
        parameter_scope: str,
        format: str = "",
        component: str = None,
    ):
        p = sf.StaticProgramParameters(self.sysman.ls, parameter_scope=parameter_scope)
        for path in paths:
            logging.info("Loading parameter fiber '%s' from: %s", parameter_scope, path)
            p.load(path, format=format)
        if not self.inference_parameters.get(component):
            self.inference_parameters[component] = []
        self.inference_parameters[component].append(p)

    def start(self):
        for component in self.inference_modules:
            component_modules = [
                sf.ProgramModule.parameter_provider(
                    self.sysman.ls, *self.inference_parameters.get(component, [])
                ),
                *self.inference_modules[component],
            ]
            self.inference_programs[component] = sf.Program(
                modules=component_modules,
                fiber=self.fibers[0],
                trace_execution=False,
            )

        # TODO: export vmfbs with multiple batch size entrypoints

        self.inference_functions["encode"] = {}
        for bs in self.model_params.clip_batch_sizes:
            self.inference_functions["encode"][bs] = self.inference_programs["clip"][
                f"{self.model_params.clip_module_name}.encode_prompts"
            ]

        self.inference_functions["denoise"] = {}
        for bs in self.model_params.unet_batch_sizes:
            self.inference_functions["denoise"][bs] = {
                "unet": self.inference_programs["unet"][
                    f"{self.model_params.unet_module_name}.run_forward"
                ],
                "init": self.inference_programs["scheduler"][
                    f"{self.model_params.scheduler_module_name}.run_initialize"
                ],
                "scale": self.inference_programs["scheduler"][
                    f"{self.model_params.scheduler_module_name}.run_scale"
                ],
                "step": self.inference_programs["scheduler"][
                    f"{self.model_params.scheduler_module_name}.run_step"
                ],
            }

        self.inference_functions["decode"] = {}
        for bs in self.model_params.vae_batch_sizes:
            self.inference_functions["decode"][bs] = self.inference_programs["vae"][
                f"{self.model_params.vae_module_name}.decode"
            ]

        self.batcher.launch()

    def shutdown(self):
        self.batcher.shutdown()

    def __repr__(self):
        return (
            f"ServiceManager(\n"
            f"  model_params={self.model_params}\n"
            f"  inference_modules={self.inference_modules}\n"
            f"  inference_parameters={self.inference_parameters}\n"
            f")"
        )


########################################################################################
# Batcher
########################################################################################


class BatcherProcess(sf.Process):
    """The batcher is a persistent process responsible for flighting incoming work
    into batches.
    """

    STROBE_SHORT_DELAY = 0.1
    STROBE_LONG_DELAY = 0.25

    def __init__(self, service: GenerateService):
        super().__init__(fiber=service.fibers[0])
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
        super().__init__(fiber=service.fibers[0])
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
            for tokenizer in self.service.tokenizers:
                for prompt in prompts:
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

        entrypoints = self.service.inference_functions["encode"]
        for bs, fn in entrypoints.items():
            if bs >= req_bs:
                break

        # Prepare tokenized input ids for CLIP inference

        clip_inputs = [
            sfnp.device_array.for_device(
                device, [req_bs, self.service.model_params.max_seq_len], sfnp.sint64
            )
        ] * 4
        for idx, arr in enumerate(clip_inputs):
            host_arr = arr.for_transfer()
            for i in range(req_bs):
                with host_arr.view(i).map(write=True, discard=True) as m:

                    # TODO: fix this attr redundancy
                    np_arr = requests[i].input_ids[idx].input_ids[0]

                    m.fill(np_arr)
            arr.copy_from(host_arr)

        # Encode tokenized inputs.
        logger.info(
            "INVOKE %r: %s",
            fn,
            "".join([f"\n  {i}: {ary.shape}" for i, ary in enumerate(clip_inputs)]),
        )
        await device
        pe, te = await fn(*clip_inputs)

        await device
        for i in range(req_bs):
            requests[i].prompt_embeds = pe.view(i)
            requests[i].add_text_embeds = te.view(i)

        return

    async def _denoise(self, device, requests):
        req_bs = len(requests)
        # Produce denoised latents
        entrypoints = self.service.inference_functions["denoise"]
        for bs, fns in entrypoints.items():
            if bs >= req_bs:
                break

        # fns is a dict of denoising components
        return

    async def _decode(self, device, requests):
        req_bs = len(requests)
        # Decode latents to images
        entrypoints = self.service.inference_functions["decode"]
        for bs, fn in entrypoints.items():
            if bs >= req_bs:
                break

        # do something with vae

        return

    async def _postprocess(self, device, requests):
        # Process output images
        for req in requests:
            req.result_image = Image.open("sample.png", mode="r").tobytes()
        return
