# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import logging
import math
import numpy as np
from tqdm.auto import tqdm
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
                    f"{self.model_params.unet_module_name}.{self.model_params.unet_fn_name}"
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
            neg_ids_list = []
            for tokenizer in self.service.tokenizers:
                input_ids = tokenizer.encode(request.prompt)
                input_ids_list.append(input_ids)
                neg_ids = tokenizer.encode(request.neg_prompt)
                neg_ids_list.append(neg_ids)
            ids_list = [*input_ids_list, *neg_ids_list]

            request.input_ids = ids_list

            # Generate random sample latents.
            seed = request.seed
            channels = self.service.model_params.num_latents_channels
            unet_dtype = self.service.model_params.unet_dtype
            latents_shape = (
                1,
                channels,
                request.height // 8,
                request.width // 8,
            )

            # Create and populate sample device array.
            generator = sfnp.RandomGenerator(seed)
            request.sample = sfnp.device_array.for_device(
                device, latents_shape, unet_dtype
            )

            sample_host = request.sample.for_transfer()
            with sample_host.map(discard=True) as m:
                m.fill(bytes(1))

            sfnp.fill_randn(sample_host, generator=generator)

            request.sample.copy_from(sample_host)
            await device
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
            ),
            sfnp.device_array.for_device(
                device, [req_bs, self.service.model_params.max_seq_len], sfnp.sint64
            ),
            sfnp.device_array.for_device(
                device, [req_bs, self.service.model_params.max_seq_len], sfnp.sint64
            ),
            sfnp.device_array.for_device(
                device, [req_bs, self.service.model_params.max_seq_len], sfnp.sint64
            ),
        ]
        host_arrs = [None] * 4
        for idx, arr in enumerate(clip_inputs):
            host_arrs[idx] = arr.for_transfer()
            for i in range(req_bs):
                with host_arrs[idx].view(i).map(write=True, discard=True) as m:

                    # TODO: fix this attr redundancy
                    np_arr = requests[i].input_ids[idx].input_ids

                    m.fill(np_arr)
            clip_inputs[idx].copy_from(host_arrs[idx])

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
            cfg_mult = 2
            requests[i].prompt_embeds = pe.view(slice(i * cfg_mult, (i + 1) * cfg_mult))
            requests[i].add_text_embeds = te.view(
                slice(i * cfg_mult, (i + 1) * cfg_mult)
            )

        return

    async def _denoise(self, device, requests):
        req_bs = len(requests)
        cfg_mult = 2 if self.service.model_params.cfg_mode else 1
        # Produce denoised latents
        entrypoints = self.service.inference_functions["denoise"]
        for bs, fns in entrypoints.items():
            if bs >= req_bs:
                break

        # Get shape of batched latents.
        # This assumes all requests are dense at this point.
        latents_shape = [
            req_bs,
            self.service.model_params.num_latents_channels,
            requests[0].height // 8,
            requests[0].width // 8,
        ]
        # Assume we are doing classifier-free guidance
        hidden_states_shape = [
            req_bs * cfg_mult,
            self.service.model_params.max_seq_len,
            2048,
        ]
        text_embeds_shape = [
            req_bs * cfg_mult,
            1280,
        ]
        denoise_inputs = {
            "sample": sfnp.device_array.for_device(
                device, latents_shape, self.service.model_params.unet_dtype
            ),
            "encoder_hidden_states": sfnp.device_array.for_device(
                device, hidden_states_shape, self.service.model_params.unet_dtype
            ),
            "text_embeds": sfnp.device_array.for_device(
                device, text_embeds_shape, self.service.model_params.unet_dtype
            ),
            "guidance_scale": sfnp.device_array.for_device(
                device, [req_bs], self.service.model_params.unet_dtype
            ),
        }

        # Send guidance scale to device.
        gs_host = denoise_inputs["guidance_scale"].for_transfer()
        for i in range(req_bs):
            cfg_dim = i * cfg_mult
            with gs_host.view(i).map(write=True, discard=True) as m:

                np_arr = np.asarray(requests[i].guidance_scale, dtype="float16")

                m.fill(np_arr)
            # Batch sample latent inputs on device.
            req_samp = requests[i].sample
            denoise_inputs["sample"].view(i).copy_from(req_samp)

            # Batch CLIP hidden states.
            enc = requests[i].prompt_embeds
            denoise_inputs["encoder_hidden_states"].view(
                slice(cfg_dim, cfg_dim + cfg_mult)
            ).copy_from(enc)

            # Batch CLIP text embeds.
            temb = requests[i].add_text_embeds
            denoise_inputs["text_embeds"].view(
                slice(cfg_dim, cfg_dim + cfg_mult)
            ).copy_from(temb)

        denoise_inputs["guidance_scale"].copy_from(gs_host)

        await device
        # Initialize scheduler.
        logger.info(
            "INVOKE %r: %s",
            fns["init"],
            "".join([f"\n  0: {latents_shape}"]),
        )
        (latents, time_ids, step_indexes, timesteps,) = await fns[
            "init"
        ](denoise_inputs["sample"])
        await device
        ts_host = timesteps.for_transfer()
        ts_host.copy_from(timesteps)
        for i, t in tqdm(
            enumerate(ts_host.items),
        ):
            step = sfnp.device_array.for_device(device, [1], sfnp.sint64)
            s_host = step.for_transfer()
            with s_host.map(write=True) as m:
                s_host.items = [i]
            step.copy_from(s_host)
            scale_inputs = [
                latents,
                step,
                timesteps,
            ]
            logger.info(
                "INVOKE %r: %s",
                fns["scale"],
                "".join(
                    [f"\n  {i}: {ary.shape}" for i, ary in enumerate(scale_inputs)]
                ),
            )
            await device
            latent_model_input, t = await fns["scale"](*scale_inputs)
            await device

            unet_inputs = [
                latent_model_input,
                t,
                denoise_inputs["encoder_hidden_states"],
                denoise_inputs["text_embeds"],
                time_ids,
                denoise_inputs["guidance_scale"],
            ]
            logger.info(
                "INVOKE %r: %s",
                fns["unet"],
                "".join([f"\n  {i}: {ary.shape}" for i, ary in enumerate(unet_inputs)]),
            )
            await device
            (noise_pred,) = await fns["unet"](*unet_inputs)
            await device

            step_inputs = [noise_pred, t, latents]
            logger.info(
                "INVOKE %r: %s",
                fns["step"],
                "".join([f"\n  {i}: {ary.shape}" for i, ary in enumerate(step_inputs)]),
            )
            await device
            (latent_model_output,) = await fns["step"](*step_inputs)
            latents.copy_from(latent_model_output)
            await device

        for idx, req in enumerate(requests):
            req.denoised_latents = sfnp.device_array.for_device(
                device, latents_shape, self.service.model_params.vae_dtype
            )
            req.denoised_latents.copy_from(latents.view(idx))
        await device
        return

    async def _decode(self, device, requests):
        req_bs = len(requests)
        # Decode latents to images
        entrypoints = self.service.inference_functions["decode"]
        for bs, fn in entrypoints.items():
            if bs >= req_bs:
                break

        latents_shape = [
            req_bs,
            self.service.model_params.num_latents_channels,
            requests[0].height // 8,
            requests[0].width // 8,
        ]
        latents = sfnp.device_array.for_device(
            device, latents_shape, self.service.model_params.vae_dtype
        )
        for i in range(req_bs):
            latents.view(i).copy_from(requests[i].denoised_latents)

        await device
        # Decode the denoised latents.
        (image,) = await fn(latents)

        await device
        images_shape = [
            req_bs,
            3,
            requests[0].height,
            requests[0].width,
        ]
        image_shape = [
            req_bs,
            3,
            requests[0].height,
            requests[0].width,
        ]
        images_host = sfnp.device_array.for_host(device, images_shape, sfnp.float16)
        images_host.copy_from(image)
        for idx, req in enumerate(requests):
            image_array = images_host.view(idx).items
            dtype = image_array.typecode
            if images_host.dtype == sfnp.float16:
                dtype = np.float16
            req.image_array = np.frombuffer(image_array, dtype=dtype).reshape(
                *image_shape
            )
        return

    async def _postprocess(self, device, requests):
        # Process output images
        for req in requests:
            # TODO: reimpl with sfnp
            permuted = (
                torch.from_numpy(req.image_array)
                .cpu()
                .permute(0, 2, 3, 1)
                .float()
                .numpy()[0]
            )
            cast_image = (permuted * 255).round().astype("uint8")
            req.result_image = Image.fromarray(cast_image).tobytes()
        return
