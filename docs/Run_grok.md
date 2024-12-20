# Run Grok-1 (Q4_1)

## Setup sharktank/shortfin:

Follow: https://github.com/nod-ai/shark-ai/blob/main/docs/developer_guide.md

## Download artifacts:

If using MI300X-3 system, irpa/mlir/tokenizer files can be found under `/data/grok/` <br>
[Instructions](https://confluence.amd.com/pages/viewpage.action?spaceKey=ENGIT&title=Nod.AI+Lab#Nod.AILab-MI300NodAIMachines) to login.

To download locally:
* Get SAS token from Azure following these [steps](https://github.com/nod-ai/llm-dev/blob/main/llama_benchmarking.md#1-get-the-unsharded-irpa-files)

* Download the Grok irpa & tokenizer files

```bash
azcopy copy \
'https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/grok-1/grok-1-q4_1.irpa?[Add SAS token here]' \
'grok-1-q4_1.irpa'
```

```bash
azcopy copy \
'https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/grok-1/tokenizer_config.json?[Add SAS token here]' \
'tokenizer_config.json'
```

```bash
azcopy copy \
'https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/grok-1/tokenizer.json?[Add SAS token here]' \
'tokenizer.json'
```

### Run Grok in sharktank with torch:

Takes ~30 mins to load the model + run prefill (generate kv cache)

This command isn't interactive. For a given prompt, it continues to predict next tokens until stopped.

```bash
python -m sharktank.examples.paged_llm_v1  \
  --gguf-file=/data/grok/grok-1-q4_1.irpa \
  --tokenizer-config-json=/data/grok/tokenizer_config.json \
  "I can assure you that the meaning of life and the universe is none other than"
```

### Export to mlir

```bash
python -m sharktank.examples.export_paged_llm_v1 \
  --irpa-file=/data/grok/grok-1-q4_1.irpa \
  --output-mlir=grok-1-q4_1.mlir \
  --output-config=grok-1-q4_1.json
```

### Compile to vmfb
This step fails with [compile error](https://gist.github.com/archana-ramalingam/aa2f26256d768051aa46f4ded5a20e14)

```bash
../iree-build/tools/iree-compile grok-1-q4_1.mlir \
  --iree-hal-target-backends=rocm \
  --iree-hip-target=gfx942 \
  -o grok-1-q4_1.vmfb
```
