# LLM Evaluation Pipeline

## Setup
Setup SHARK Platform's Evaluation Pipeline

```
pip install -r sharktank/requirements-tests.txt
```

### Perplexity

Test perplexity for Llama3.1 8B & 405B (FP16 & FP8) models:

```bash
pytest sharktank/tests/evaluate/perplexity_test.py  --longrun
```

Get perplexity for a new model:

```bash
python -m  sharktank.evaluate.perplexity \
  --gguf-file=llama3_70b_f16.gguf \
  --tokenizer-config-json=tokenizer_config.json
```
