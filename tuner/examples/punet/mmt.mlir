// RUN: iree-compile --iree-hal-target-backends=rocm --iree-rocm-target-chip=gfx942 \
// RUN:   --iree-rocm-link-bc=true --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode \
// RUN:   --iree-global-opt-propagate-transposes=true --iree-opt-outer-dim-concat=true \
// RUN:   --iree-opt-const-eval=false --iree-codegen-gpu-native-math-precision=true --iree-rocm-waves-per-eu=2 \
// RUN:   --iree-preprocessing-pass-pipeline='builtin.module(iree-preprocessing-transpose-convolution-pipeline)' \
// RUN:   --iree-codegen-llvmgpu-use-vector-distribution --iree-codegen-transform-dialect-library=config.mlir \
// RUN:   %s -o %s.vmfb

// To compile to for benchmarking, add:
//  --iree-flow-export-benchmark-funcs --iree-hal-benchmark-dispatch-repeat-count=1000
//
// To benchmark:
//   for i in {0..4} ; do
//     iree-benchmark-module --device=rocm://7 --module=%s.vmfb --function="main_${i}_benchmark" --device_allocator=caching \
//       --batch_size=1000 --benchmark_repetitions=5
//   done

!matA_0 = tensor<2048x1280xf16>
!matB_0 = tensor<10240x1280xf16>
!matC_0 = tensor<2048x10240xf32>

func.func @main_0(%arg0: !matA_0, %arg1: !matB_0) -> !matC_0 {
  %cst = arith.constant 0.000000e+00 : f16
  %5 = tensor.empty() : !matC_0
  %6 = linalg.fill ins(%cst : f16) outs(%5 : !matC_0) -> !matC_0
  %8 = linalg.matmul_transpose_b ins(%arg0, %arg1 : !matA_0, !matB_0) outs(%6 : !matC_0) -> !matC_0
  return %8 : !matC_0
}

!matA_1 = tensor<2048x1280xf16>
!matB_1 = tensor<1280x1280xf16>
!matC_1 = tensor<2048x1280xf32>

func.func @main_1(%arg0: !matA_1, %arg1: !matB_1) -> !matC_1 {
  %cst = arith.constant 0.000000e+00 : f16
  %5 = tensor.empty() : !matC_1
  %6 = linalg.fill ins(%cst : f16) outs(%5 : !matC_1) -> !matC_1
  %8 = linalg.matmul_transpose_b ins(%arg0, %arg1 : !matA_1, !matB_1) outs(%6 : !matC_1) -> !matC_1
  return %8 : !matC_1
}

!matA_2 = tensor<2048x5120xf16>
!matB_2 = tensor<1280x5120xf16>
!matC_2 = tensor<2048x1280xf32>

func.func @main_2(%arg0: !matA_2, %arg1: !matB_2) -> !matC_2 {
  %cst = arith.constant 0.000000e+00 : f16
  %5 = tensor.empty() : !matC_2
  %6 = linalg.fill ins(%cst : f16) outs(%5 : !matC_2) -> !matC_2
  %8 = linalg.matmul_transpose_b ins(%arg0, %arg1 : !matA_2, !matB_2) outs(%6 : !matC_2) -> !matC_2
  return %8 : !matC_2
}

!matA_3 = tensor<128x2048xf16>
!matB_3 = tensor<1280x2048xf16>
!matC_3 = tensor<128x1280xf32>

func.func @main_3(%arg0: !matA_3, %arg1: !matB_3) -> !matC_3 {
  %cst = arith.constant 0.000000e+00 : f16
  %5 = tensor.empty() : !matC_3
  %6 = linalg.fill ins(%cst : f16) outs(%5 : !matC_3) -> !matC_3
  %8 = linalg.matmul_transpose_b ins(%arg0, %arg1 : !matA_3, !matB_3) outs(%6 : !matC_3) -> !matC_3
  return %8 : !matC_3
}

!matA_4 = tensor<8192x640xf16>
!matB_4 = tensor<5120x640xf16>
!matC_4 = tensor<8192x5120xf32>

func.func @main_4(%arg0: !matA_4, %arg1: !matB_4) -> !matC_4 {
  %cst = arith.constant 0.000000e+00 : f16
  %5 = tensor.empty() : !matC_4
  %6 = linalg.fill ins(%cst : f16) outs(%5 : !matC_4) -> !matC_4
  %8 = linalg.matmul_transpose_b ins(%arg0, %arg1 : !matA_4, !matB_4) outs(%6 : !matC_4) -> !matC_4
  return %8 : !matC_4
}
