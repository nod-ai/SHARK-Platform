# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import argparse
import numpy
import sys

from transformers import LlamaTokenizer  # type: ignore

from iree.runtime import (  # type: ignore
    HalElementType,
)

from shortfin.framework.session import DeviceSession

from shortfin.llm.attn_block_cache import (
    create_attn_block_cache_module,
    AttnBlockCache,
)

from shortfin.llm.config import (
    CacheParams,
    ModelParams,
    ServiceParams,
)

from shortfin.llm.impl.service_v1 import GenerateServiceV1
from shortfin.llm.service import GenerateRequest


def setup(vmfb_path, config_path, gguf_path):
    from iree.runtime._binding import disable_leak_checker  # type: ignore

    model_params = ModelParams.load_json(config_path)

    device_block_count = model_params.max_seq_len // model_params.block_seq_stride
    cache_params = CacheParams(
        model=model_params,
        device_block_count=device_block_count,
        block_pos_stride=model_params.block_seq_stride,
    )

    disable_leak_checker()
    session = DeviceSession(uri="local-sync", queue_count=2)
    attn_block_cache = AttnBlockCache(session, cache_params)

    lms = session.create_module_set(model_params.module_name, context_count=1)
    lms.load_io_module(gguf_path)
    lms.load_vmfb(vmfb_path)
    lms.add(create_attn_block_cache_module(attn_block_cache))
    lms.initialize()

    params = ServiceParams(cache=cache_params, model=model_params)
    service = GenerateServiceV1(session=session, params=params, cache=attn_block_cache)
    return service


def map_buffer(value):
    mapped = value.map()
    return mapped.asarray(value.shape, HalElementType.map_to_dtype(value.element_type))


async def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", help="name of hugginface tokenizer to use")
    parser.add_argument("--config", help="json config file with hyperparameters")
    parser.add_argument("--vmfb", help="vmfb with compiler LLM kernels")
    parser.add_argument("--gguf", help="gguf file containing modle coefficients")
    parsed = parser.parse_args(argv)

    hf_path = parsed.tokenizer
    config_path = parsed.config
    vmfb_path = parsed.vmfb
    gguf_path = parsed.gguf

    service = setup(vmfb_path, config_path, gguf_path)
    tokenizer = LlamaTokenizer.from_pretrained(hf_path)
    state = service.start()

    for line in ["one two three four five six seven eight"]:
        prompt = line.strip()
        if not prompt:
            break

        input_ids = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
        print(input_ids)
        request = GenerateRequest("request_id", prompt, input_ids)
        await state.set_sequences([request])
        logits = await state.prefill()

        seq_len = len(input_ids)
        mapped_logits = map_buffer(logits.value)
        predicted_tokens = numpy.argmax(mapped_logits[0, :seq_len], axis=-1)
        predicted_token = predicted_tokens[-1]
        print(f"Predicted token: {predicted_token}")

        # TODO Determine why decode step crashes
        # await state.set_decode_step([predicted_token])
        # logits = await state.decode()
        # mapped_logits = map_buffer(logits)
        # print(numpy.argmax(mapped_logits, axis=-1))
        await state.recycle()

    service.shutdown()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
