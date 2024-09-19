import torch
from sharktank.examples.paged_llm_v1 import *
from sharktank.utils import tokenizer
from sharktank.utils import hf_datasets
import unittest
from pathlib import Path

default_arguments = {
    "hf_dataset": "llama3_8B_f16",
    "tokenizer-config-json": Path("./tokenizer_config.json"),
    "prompt": ["I believe the meaning of life is"],
    "device": None,
    "activation-dtype": "float32",
}


class Llama8BTest(unittest.TestCase):
    def setUp(self):
        self.device = (
            torch.device(default_arguments["device"])
            if default_arguments["device"]
            else None
        )
        self.activation_dtype = getattr(torch, default_arguments["activation-dtype"])
        assert isinstance(self.activation_dtype, torch.dtype)
        self.data_files = hf_datasets.get_dataset(
            default_arguments["hf_dataset"]
        ).download(local_dir=Path("."))
        self.dataset = Dataset.load(self.data_files["gguf"], file_type="gguf")
        self.tokenizer_config = tokenizer.load_tokenizer(
            default_arguments["tokenizer-config-json"].parent,
            tokenizer_type="transformers",
        )
        self.prompts = default_arguments["prompt"]

    def createConfigModel(self, kv_cache_type):
        return LlamaModelConfig(
            hp=configs.LlamaHParams.from_gguf_props(self.dataset.properties),
            block_seq_stride=16,
            kv_cache_type=kv_cache_type,
            device=self.device,
            activation_dtype=self.activation_dtype,
            attention_dtype=self.activation_dtype,
        )

    def runPrefill(self, kv_cache_type):
        config = self.createConfigModel(kv_cache_type)
        model = PagedLlamaModelV1(self.dataset.root_theta, config)
        generator = TorchGenerator(model, self.tokenizer_config)
        batch = generator.begin_batch(self.prompts)
        batch.prefill()

        llama_cpp_8b_prefill_token = [[311]]
        llama_cpp_8b_prefill_token_logit = torch.tensor(15.613972)

        bs, *_ = batch.logits.shape
        assert len(batch.seq_lens) == bs
        greedy_token_logit = 0.0
        for b, seq_len in enumerate(batch.seq_lens):
            step_logits = batch.logits[b, seq_len - 1]
            greedy_token_logit = step_logits[torch.argmax(step_logits)]

        rtol = 3e-4
        atol = 4e-3
        assert batch.results == llama_cpp_8b_prefill_token
        torch.testing.assert_close(
            greedy_token_logit, llama_cpp_8b_prefill_token_logit, rtol=rtol, atol=atol
        )

    def testPrefillPaged(self):
        self.runPrefill("paged")

    def testPrefillDirect(self):
        self.runPrefill("direct")


if __name__ == "__main__":
    unittest.main()
