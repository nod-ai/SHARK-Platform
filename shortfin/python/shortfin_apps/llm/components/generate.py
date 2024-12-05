# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import io
import logging
from pprint import pformat

import shortfin as sf
import shortfin.array as sfnp

# TODO: Have a generic "Responder" interface vs just the concrete impl.
from shortfin.interop.fastapi import FastAPIResponder

from .io_struct import GenerateReqInput
from .messages import InferenceExecRequest, InferencePhase
from .service import GenerateService
from .tokenizer import Encoding

logger = logging.getLogger(__name__)


class GenerateItemProcess(sf.Process):
    """Process instantiated for each generation sequence.

    This process breaks the sequence into individual inference and sampling
    steps, submitting them to the batcher and marshaling incremental/final
    results.
    """

    def __init__(
        self,
        client: "ClientGenerateBatchProcess",
        gen_req: GenerateReqInput,
        index: int,
        input_token_ids: list[int],
        max_completion_tokens: int,
        eos_token_id: int,
    ):
        super().__init__(fiber=client.fiber)
        self.client = client
        self.gen_req = gen_req
        self.index = index
        self.input_token_ids = input_token_ids
        self.result_token_ids: list[int] = []
        self.max_completion_tokens = max_completion_tokens
        self.eos_token_id = eos_token_id

    async def run(self):
        exec = InferenceExecRequest(InferencePhase.PREFILL, self.input_token_ids)
        try:
            self.client.batcher.submit(exec)
            await exec.done

            # Prefill result.
            token = sfnp.argmax(exec.result_logits)
            token_int = token.items[0]

            self.append_token(token_int)
            # Decode loop.
            exec.start_position = len(self.input_token_ids) - 1
            for i in range(self.max_completion_tokens):
                exec.reset(InferencePhase.DECODE)
                exec.input_token_ids.append(token_int)
                exec.start_position += 1
                logger.info(
                    f"======== SNB GenerateItemD input_ids: ========\n{exec.input_token_ids}\n================"
                )
                self.client.batcher.submit(exec)
                await exec.done
                logger.info(
                    f"======== SNB GenerateItemD result_logits: ========\n{exec.result_logits}\n================"
                )
                token = sfnp.argmax(exec.result_logits)
                token_int = token.items[0]
                logger.info(
                    f"======== SNB GenerateItemD Token_int: ========\n{token_int}\n================"
                )
                self.append_token(token_int)
                logger.info(
                    f"======== SNB GenerateItemD result_ids: ========\n{self.result_token_ids}\n================"
                )
                if token_int == self.eos_token_id:
                    break
        finally:
            exec.free_cache_pages()

    def append_token(self, token: int):
        self.result_token_ids.append(token)
        self.client.stream_results(self)


class ClientGenerateBatchProcess(sf.Process):
    """Process instantiated for handling a batch from a client.

    This takes care of several responsibilities:

    * Tokenization / Detokenization
    * Splitting the batch into GenerateItemProcesses
    * Streaming responses
    * Final responses
    """

    __slots__ = [
        "batcher",
        "complete_infeed",
        "gen_req",
        "responder",
        "tokenizer",
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
        self.tokenizer = service.tokenizer
        self.batcher = service.batcher
        self.complete_infeed = self.system.create_queue()

    async def run(self):
        logger.debug("Started ClientBatchGenerateProcess: %r", self)
        streaming = self.gen_req.stream
        if streaming:
            self.responder.start_response()

        try:
            # Launch all individual generate processes and wait for them to finish.
            gen_processes = []
            # TODO: We should send this to an executor and await the results.
            input_batch = self.tokenize()
            try:
                encodings = [input_encoding.ids for input_encoding in input_batch]
                formatted_encodings = pformat(encodings, width=120, compact=True)
                logger.info(
                    f"======== SNB Input Batch ========:\n{formatted_encodings}\n======== END ========"
                )
            except Exception as e:
                logger.error(f"Error in first log: {e!r}")
            for index, input_tokens in enumerate(input_batch):
                gen_process = GenerateItemProcess(
                    self,
                    self.gen_req,
                    index,
                    input_tokens.ids,
                    max_completion_tokens=self.gen_req.sampling_params[
                        "max_completion_tokens"
                    ],
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                gen_processes.append(gen_process)
                gen_process.launch()

            await asyncio.gather(*gen_processes)

            if streaming:
                logger.debug("Responding to streaming batch")
                self.responder.stream_part(b"data: [DONE]\n\n")
                self.responder.stream_part(None)
            else:
                logging.debug("Responding to one shot batch")
                out = io.BytesIO()
                result_texts = self.tokenizer.decode(
                    [p.result_token_ids for p in gen_processes]
                )
                raw_encodings = [p.result_token_ids for p in gen_processes]
                try:
                    for i, result_text in enumerate(result_texts):
                        out.write(b"data: ")
                        formatted_encodings = pformat(
                            raw_encodings, width=120, compact=True
                        )
                        logger.info(
                            f"======== SNB Result ======== {i}:\n{formatted_encodings}\n======== END ========"
                        )
                        out.write(result_text.encode())
                        out.write(b"\n\n")
                    self.responder.send_response(out.getvalue())
                except Exception as e:
                    logger.error(f"Error in second log: {e!r}")
        finally:
            self.responder.ensure_response()
            logger.info(
                f"\n\n================ Done with {str(self.gen_req.rid)} ================\n\n\n\n"
            )

    def stream_results(self, gen_process: GenerateItemProcess):
        if not self.gen_req.stream:
            return
        (result_text,) = self.tokenizer.decode([gen_process.result_token_ids])
        out = io.BytesIO()
        out.write(b"data: ")
        out.write(result_text.encode())
        out.write(b"\n\n")
        self.responder.stream_part(out.getvalue())

    def tokenize(self) -> list[Encoding]:
        gen_req = self.gen_req
        if gen_req.text is not None:
            if self.gen_req.is_single:
                texts = [self.gen_req.text]
                logger.debug("Encoding single request")
            else:
                texts = self.gen_req.text
                logger.debug("Encoding batch of %d", len(texts))
            encodings = self.tokenizer.encode(texts)
            logger.debug("Generated encodings: %r", encodings)
            return encodings
        else:
            raise NotImplementedError("GenerateReqInput.input_ids handling NYI")
