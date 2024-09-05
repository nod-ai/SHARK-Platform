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
