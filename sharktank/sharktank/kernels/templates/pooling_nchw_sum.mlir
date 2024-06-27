// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!dtype = {{dtype}}
!dynamic_tensor_type = tensor<?x?x?x?x!dtype>
!weights_tensor_type = tensor<{{weights_H}}x{{weights_W}}x!dtype>
!out_tensor_type = !dynamic_tensor_type

module {

util.func private @sharktank_pooling_nchw_sum_{{weights_H}}_{{weights_W}}_{{strides_H}}_{{strides_W}}_{{padding_H}}_{{padding_W}}_{{dilations_H}}_{{dilations_W}}_{{dtype}} (
    %input: !dynamic_tensor_type)
    -> !out_tensor_type {
  %zero = arith.constant 0: !dtype
  %weights = tensor.empty() : !weights_tensor_type
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c2 = arith.constant 2: index
  %c3 = arith.constant 3: index

  %input_pad = tensor.pad %input low[0, 0, {{padding_H}}, {{padding_W}}] high[0, 0, {{padding_H}}, {{padding_W}}] {
  ^bb0(%arg0 : index, %arg1 : index, %arg2: index, %arg3: index):
    tensor.yield %zero : !dtype
  } : !dynamic_tensor_type to !dynamic_tensor_type

  // Pooling size math, equivalent to:
  // h_out = math.floor((h + 2 * padding[0] - weights_size[0]) / strides[0] + 1)
  %iH = tensor.dim %input, %c2 : !dynamic_tensor_type
  %kH = arith.constant {{weights_H}} : index // weights_size[0]
  %sH = arith.constant {{strides_H}} : index // strides[0]
  %dH = arith.constant {{dilations_H}} : index
  %pH = arith.constant {{padding_H}} : index // padding[0]

  %rH_0 = arith.muli %pH, %c2 : index
  %rH_1 = arith.addi %iH, %rH_0 : index
  %rH_2 = arith.subi %rH_1, %kH : index
  %rH_3 = arith.addi %rH_2, %sH : index
  %rH   = arith.divsi %rH_3, %sH : index

  // w_out = math.floor((w + 2 * padding[1] - weights_size[1]) / strides[1] + 1)
  %iW = tensor.dim %input, %c3 : !dynamic_tensor_type
  %kW = arith.constant {{weights_H}} : index // weights_size[1]
  %sW = arith.constant {{strides_W}} : index // strides[1]
  %dW = arith.constant {{dilations_W}} : index
  %pW = arith.constant {{padding_W}} : index // padding[1]

  %rW_0 = arith.muli %pW, %c2 : index
  %rW_1 = arith.addi %iW, %rW_0 : index
  %rW_2 = arith.subi %rW_1, %kW : index
  %rW_3 = arith.addi %rW_2, %sW : index
  %rW   = arith.divsi %rW_3, %sW : index

  %rN = tensor.dim %input, %c0 : !dynamic_tensor_type
  %rC = tensor.dim %input, %c1 : !dynamic_tensor_type
  %result_empty = tensor.empty(%rN, %rC, %rH, %rW) : !out_tensor_type
  %result_fill = linalg.fill ins(%zero: !dtype) outs(%result_empty: !out_tensor_type) -> !out_tensor_type
  %result = linalg.pooling_nchw_sum {dilations = dense<[{{dilations_H}}, {{dilations_W}}]> : tensor<2xi64>, strides = dense<[{{strides_H}}, {{strides_W}}]> : tensor<2xi64>} ins(%input_pad, %weights: !dynamic_tensor_type, !weights_tensor_type) outs(%result_fill: !out_tensor_type) -> !out_tensor_type
  util.return %result : !out_tensor_type
}

}
