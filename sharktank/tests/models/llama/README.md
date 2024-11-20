# How to run Llama 3.1 Benchmarking Tests
In order to run Llama 3.1 8B F16 Decomposed test:
```
pytest sharktank/tests/models/llama/benchmark_amdgpu_test.py -v -s \
    --run-quick-test --iree-hip-target=gfx942
```

In order to filter by test, use the -k option. If you
wanted to only run the Llama 3.1 70B F16 Decomposed test:
```
pytest sharktank/tests/models/llama/benchmark_amdgpu_test.py -v -s \
    --run-nightly-llama-tests --iree-hip-target=gfx942 \
    -k 'testBenchmark70B_f16_TP8_Decomposed'
```
