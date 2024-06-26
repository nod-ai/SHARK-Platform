// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1)>

!dtype = {{dtype}}
!dynamic_tensor_type = tensor<?x?x?x?x!dtype>
!out_tensor_type = !dynamic_tensor_type

module {

util.func private @sharktank_conv_2d_nchw_fchw_{{strides_H}}_{{strides_W}}_{{padding_H}}_{{padding_W}}_{{dilations_H}}_{{dilations_W}}_{{dtype}} (
    %input: !dynamic_tensor_type, %input_pad: !dynamic_tensor_type, %weights: !dynamic_tensor_type, %bias: tensor<?x!dtype>)
    -> !out_tensor_type {
  %zero = arith.constant 0: !dtype
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c2 = arith.constant 2: index
  %c3 = arith.constant 3: index

  // Convolution size math, equivalent to:
  // h_out = math.floor((h + 2 * padding[0] - dilations[0] * (k0 - 1) - 1) / strides[0] + 1)
  // w_out = math.floor((w + 2 * padding[1] - dilations[1] * (k1 - 1) - 1) / strides[1] + 1)
  %iH = tensor.dim %input, %c2 : !dynamic_tensor_type
  %kH = tensor.dim %weights, %c2 : !dynamic_tensor_type
  %sH = arith.constant {{strides_H}} : index
  %dH = arith.constant {{dilations_H}} : index
  %pH = arith.constant {{padding_H}} : index

  %rH_0 = arith.subi %kH, %c1 : index
  %rH_1 = arith.muli %dH, %rH_0 : index
  %rH_2 = arith.muli %pH, %c2 : index
  %rH_3 = arith.addi %iH, %rH_2 : index
  %rH_4 = arith.subi %rH_3, %rH_1 : index
  %rH_5 = arith.subi %rH_4, %c1 : index
  %rH_6 = arith.addi %rH_5, %sH : index
  %rH   = arith.divsi %rH_6, %sH : index

  %iW = tensor.dim %input, %c3 : !dynamic_tensor_type
  %kW = tensor.dim %weights, %c3 : !dynamic_tensor_type
  %sW = arith.constant {{strides_W}} : index
  %dW = arith.constant {{dilations_W}} : index
  %pW = arith.constant {{padding_W}} : index

  %rW_0 = arith.subi %kW, %c1 : index
  %rW_1 = arith.muli %dW, %rW_0 : index
  %rW_2 = arith.muli %pW, %c2 : index
  %rW_3 = arith.addi %iW, %rW_2 : index
  %rW_4 = arith.subi %rW_3, %rW_1 : index
  %rW_5 = arith.subi %rW_4, %c1 : index
  %rW_6 = arith.addi %rW_5, %sW : index
  %rW   = arith.divsi %rW_6, %sW : index

  %rN = tensor.dim %input, %c0 : !dynamic_tensor_type
  %rC = tensor.dim %weights, %c0 : !dynamic_tensor_type
  %result_empty = tensor.empty(%rN, %rC, %rH, %rW) : !out_tensor_type
  %result_fill = linalg.fill ins(%zero: !dtype) outs(%result_empty: !out_tensor_type) -> !out_tensor_type
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<[{{dilations_H}}, {{dilations_W}}]> : tensor<2xi64>, strides = dense<[{{strides_H}}, {{strides_W}}]> : tensor<2xi64>} ins(%input_pad, %weights: !dynamic_tensor_type, !dynamic_tensor_type) outs(%result_fill: !out_tensor_type) -> !out_tensor_type
  %result_biased = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%result, %bias : !dynamic_tensor_type, tensor<?x!dtype>) outs(%result : !dynamic_tensor_type) {
    ^bb0(%in: !dtype, %in_1: !dtype, %out: !dtype):
      %add = arith.addi %in, %in_1 : !dtype
      linalg.yield %add : !dtype
    } -> !dynamic_tensor_type
  util.return %result_biased : !out_tensor_type
}

}
