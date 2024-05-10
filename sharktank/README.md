# SHARK Tank

**WARNING: This is an early preview that is in progress. It is not ready for
general use.**

Light weight inference optimized layers and models for popular genai
applications.

This sub-project is a work in progress. It is intended to be a repository of
layers, model recipes, and conversion tools from popular LLM quantization
tooling.

## Examples

The repository will ultimately grow a curated set of models and tools for
constructing them, but for the moment, it largely contains some CLI exmaples.
These are all under active development and should not yet be expected to work.


### Perform batched inference in PyTorch on a paged llama derived LLM:

```shell
python -m sharktank.examples.paged_llm_v1 \
  --hf-dataset=open_llama_3b_v2_f16_gguf \
  "Prompt 1" \
  "Prompt 2" ...
```

### Export an IREE compilable batched LLM for serving:

```shell
python -m sharktank.examples.export_paged_llm_v1 \
  --hf-dataset=open_llama_3b_v2_f16_gguf \
  --output_mlir=/tmp/open_llama_3b_v2_f16.mlir \
  --output_config=/tmp/open_llama_3b_v2_f16.json
```

### Dump parsed information about a model from a gguf file:

```shell
python -m sharktank.tools.dump_gguf --hf-dataset=open_llama_3b_v2_f16_gguf
```
