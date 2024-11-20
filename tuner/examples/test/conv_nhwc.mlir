!convA_0 = tensor<2x34x34x1280xi8>
!convB_0 = tensor<3x3x1280x1280xi8>
!convC_0 = tensor<2x32x32x1280xi32>

func.func @main_0(%arg0: !convA_0, %arg1: !convB_0) -> !convC_0 {
  %cst = arith.constant 0.000000e+00 : f16
  %5 = tensor.empty() : !convC_0
  %6 = linalg.fill ins(%cst : f16) outs(%5 : !convC_0) -> !convC_0
  %8 = linalg.conv_2d_nhwc_hwcf ins(%arg0, %arg1 : !convA_0, !convB_0) outs(%6 : !convC_0) -> !convC_0
  return %8 : !convC_0
}
