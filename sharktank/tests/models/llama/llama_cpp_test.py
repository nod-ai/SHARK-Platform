import torch
from sharktank.examples.paged_llm_v1 import *
from sharktank.utils import tokenizer
import unittest
from pathlib import Path
from copy import deepcopy

default_arguments = {
    "tokenizer-config-json": Path(
        "/home/avsharma/Llama-2-7b-chat-hf/tokenizer_config.json"
    ),
    "prompt": ["I believe the meaning of life is"],
    "kv-cache-type": "direct",
    "device": None,
    "activation-dtype": "float32",
}


class Llama2Test(unittest.TestCase):
    def test7BModel(self):
        device = (
            torch.device(default_arguments["device"])
            if default_arguments["device"]
            else None
        )
        activation_dtype = getattr(torch, default_arguments["activation-dtype"])
        assert isinstance(activation_dtype, torch.dtype)
        dataset = Dataset.load("../Llama-2-7B-f16.gguf", file_type="gguf")
        tokenizer_config = tokenizer.load_tokenizer(
            default_arguments["tokenizer-config-json"].parent,
            tokenizer_type="transformers",
        )
        prompts = default_arguments["prompt"]
        config_direct = LlamaModelConfig(
            hp=configs.LlamaHParams.from_gguf_props(dataset.properties),
            block_seq_stride=16,
            kv_cache_type=default_arguments["kv-cache-type"],
            device=device,
            activation_dtype=activation_dtype,
            attention_dtype=activation_dtype,
        )
        config_paged = deepcopy(config_direct)
        config_paged.kv_cache_type = "paged"
        model_direct = PagedLlamaModelV1(dataset.root_theta, config_direct)
        model_paged = PagedLlamaModelV1(dataset.root_theta, config_paged)
        generator_direct = TorchGenerator(model_direct, tokenizer_config)
        generator_paged = TorchGenerator(model_paged, tokenizer_config)

        batch_direct = generator_direct.begin_batch(prompts)
        batch_direct.prefill()

        while not batch_direct.done:
            batch_direct.decode()

        batch_paged = generator_paged.begin_batch(prompts)
        batch_paged.prefill()

        while not batch_paged.done:
            batch_paged.decode()

        assert batch_direct.results == batch_paged.results
