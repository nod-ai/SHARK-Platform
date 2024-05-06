# Model cookbook

Note: These are early commands that the sharktank team is using and will turn
into proper docs later.

## Useful tools and projects

* https://huggingface.co/docs/huggingface_hub/en/guides/cli
* https://github.com/ggerganov/llama.cpp (specifically for [`convert.py`](https://github.com/ggerganov/llama.cpp/blob/master/convert.py))

## Create a non-quantized .gguf file

Models on Hugging Face in the GGUF format are sometimes only uploaded with
certain quantized (e.g. 8 bit or lower) types. For example,
https://huggingface.co/SlyEcho/open_llama_3b_v2_gguf has these types:
`[f16, q4_0, q4_1, q5_0, q5_1, q8_0]`.

To convert our own Llama-3-8B F16 GGUF, we can find a source model (i.e.
safetensors or PyTorch) like https://huggingface.co/NousResearch/Meta-Llama-3-8B
and use the following commands:

```bash
huggingface-cli download --local-dir . NousResearch/Meta-Llama-3-8B

python ~/llama.cpp/convert.py --outtype f16 --outfile Meta-Llama-3-8B-f16.gguf . --vocab-type bpe
```

Another example:

```bash
huggingface-cli login
huggingface-cli download --local-dir /tmp/mistral-7b mistralai/Mistral-7B-v0.1
python ~/llama.cpp/convert.py --outtype f32 --outfile /tmp/mistral-7b-v0.1-f32.gguf /tmp/mistral-7b

# Run through reference implementation
python -m sharktank.examples.paged_llm_v1 \
  --gguf-file=/tmp/mistral-7b-v0.1-f32.gguf \
  --tokenizer-config-json=/tmp/mistral-7b/tokenizer_config.json \
  "Prompt"

# Export as MLIR
python -m sharktank.examples.export_paged_llm_v1 \
  --gguf-file=/tmp/mistral-7b-v0.1-f32.gguf \
  --output=/tmp/mistral-7b-v0.1-f32.mlir
```

See also the documentation at
https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#prepare-and-quantize

## Using registered datasets with automatic file fetching

A number of GGUF datasets are coded directly in to the
[`hf_datasets.py`](/sharktank/sharktank/utils/hf_datasets.py) file. These can be
used with the `--hf-dataset=` flag, which will automatically fetch files using
[`hf_hub_download()`](https://huggingface.co/docs/huggingface_hub/en/guides/download).

* Note that the cache used by Hugging Face Hub can be
[customized](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache)
with the `HF_HOME` and `HF_HUB_CACHE` environment variables.

For example, to run the
[`paged_llm_v1`](/sharktank/sharktank/examples/paged_llm_v1.py) script with the
`open_llama_3b_v2_q8_0_gguf` dataset from
[SlyEcho/open_llama_3b_v2_gguf](https://huggingface.co/SlyEcho/open_llama_3b_v2_gguf):

```bash
python -m sharktank.examples.paged_llm_v1 --hf-dataset=open_llama_3b_v2_q8_0_gguf "Prompt 1"

open-llama-3b-v2-q8_0.gguf: 100%|█████████████████████████████| 3.64G/3.64G [01:35<00:00, 38.3MB/s]
tokenizer.model: 100%|███████████████████████████████████████████| 512k/512k [00:00<00:00, 128MB/s]
tokenizer_config.json: 100%|██████████████████████████████████████████████| 593/593 [00:00<?, ?B/s]
:: Prompting:
    b'Prompt 1'
:: Prompt tokens: tensor([[    1,  6874,   448, 29500, 29532,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0]])
:: Invoke prefill:
```
