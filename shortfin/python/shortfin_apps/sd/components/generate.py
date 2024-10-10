# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import io
import logging

import shortfin as sf
import shortfin.array as sfnp

# TODO: Have a generic "Responder" interface vs just the concrete impl.
from shortfin.interop.fastapi import FastAPIResponder

from .io_struct import GenerateReqInput
from .messages import InferenceExecRequest, InferencePhase
from .service import GenerateService
from .tokenizer import Encoding

logger = logging.getLogger(__name__)

class GenerateImageProcess(sf.Process):
    """Process instantiated for every image generation.

    This process breaks the sequence into individual inference and sampling
    steps, submitting them to the batcher and marshaling final
    results.

    Responsible for a single image.
    """

    def __init__(
        self,
        client: "ClientGenerateBatchProcess",
        gen_req: GenerateReqInput,
        index: int,
    ):
        super().__init__(fiber=client.fiber)
        self.client = client
        self.gen_req = gen_req
        self.index = index
        self.result_image = None

    async def run(self):
        # TODO: make sure gen_req is being indexed for the singular image generation sequence's inputs.
        exec = InferenceExecRequest(
            InferencePhase.PREPARE, 
            [self.gen_req.prompt[self.index]], 
            [self.gen_req.neg_prompt[self.index]],
            self.gen_req.height[self.index],
            self.gen_req.width[self.index],
            )
        self.client.batcher.submit(exec)
        await exec.done
        (
            exec.sample,
            exec.input_ids,
            exec.neg_input_ids,
            exec.guidance_scale,
            exec.timesteps,
            exec.time_ids,
        ) = exec.payload

        exec.reset(InferencePhase.ENCODE)
        self.client.batcher.submit(exec)
        await exec.done
        (
            exec.prompt_embeds,
            exec.neg_embeds,
        ) = exec.payload

        exec.reset(InferencePhase.DENOISE)
        self.client.batcher.submit(exec)
        await exec.done
        exec.denoised_latents = exec.payload

        exec.reset(InferencePhase.DECODE)
        self.client.batcher.submit(exec)
        await exec.done
        exec.image_array = exec.payload

        exec.reset(InferencePhase.POSTPROCESS)
        self.client.batcher.submit(exec)
        await exec.done
        self.result_image = exec.payload

class ClientGenerateBatchProcess(sf.Process):
    """Process instantiated for handling a batch from a client.

    This takes care of several responsibilities:

    * Tokenization
    * Random Latents Generation
    * Splitting the batch into GenerateImageProcesses
    * Streaming responses
    * Final responses
    """

    __slots__ = [
        "batcher",
        "complete_infeed",
        "gen_req",
        "responder",
    ]

    def __init__(
        self,
        service: GenerateService,
        gen_req: GenerateReqInput,
        responder: FastAPIResponder,
    ):
        super().__init__(fiber=service.main_fiber)
        self.gen_req = gen_req
        self.responder = responder
        self.batcher = service.batcher
        self.complete_infeed = self.system.create_queue()

    async def run(self):
        logger.debug("Started ClientBatchGenerateProcess: %r", self)
        try:
            # Launch all individual generate processes and wait for them to finish.
            gen_processes = []
            for index in range(self.gen_req.num_output_images):
                gen_process = GenerateImageProcess(
                    self, self.gen_req, index
                )
                gen_processes.append(gen_process)
                gen_process.launch()

            await asyncio.gather(*gen_processes)

            #TODO: stream image outputs
            logging.debug("Responding to one shot batch")
            out = io.BytesIO()
            result_images = [p.result_image for p in gen_processes]
            for idx, result_image in enumerate(result_images):
                out.write(f"generated image #{idx}")
                #TODO: save or return images
            self.responder.send_response(out.getvalue())
        finally:
            self.responder.ensure_response()
