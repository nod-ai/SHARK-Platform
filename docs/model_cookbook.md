Note: These are early commands that the sharktank team is using and will turn into proper docs later.

# Create a Non-Quantized .gguf file

To create a Llama-3-8B F16 gguf, we can use the following commands:

```
huggingface-cli download --local-dir . NousResearch/Meta-Llama-3-8B

python ~/llama.cpp/convert.py --outtype f16 --outfile Meta-Llama-3-8B-f16.gguf . --vocab-type bpe
```
