### How to build llama.cpp logit comparison branch
```
git clone https://github.com/aviator19941/llama.cpp.git
cd llama.cpp/
git checkout llama_comparison
cmake -B build
cmake --build build --config Release
```

### How to run llama.cpp
```
huggingface-cli download meta-llama/Meta-Llama-3.1-70B --local-dir /home/avsharma/Meta-Llama-3.1-70B
python convert_hf_to_gguf.py --outtype f16 --outfile Llama-3.1-70B-f16.gguf ../Meta-Llama-3.1-70B/
```

To predict the prefill token, use `--n-predict 1` and to predict the first decode token, use `--n-predict 2`:
```
./build/bin/llama-cli -m Llama-3.1-70B-f16.gguf -p "I believe the meaning of life is" --threads 1 --temp 0 --n-predict 1 --no-warmup
```
