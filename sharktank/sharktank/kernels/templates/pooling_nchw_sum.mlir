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
    %input_pad: !dynamic_tensor_type)
    -> !out_tensor_type {
  %zero = arith.constant 0: !dtype
  %weights = tensor.empty() : !weights_tensor_type
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c2 = arith.constant 2: index
  %c3 = arith.constant 3: index

  %rN = tensor.dim %input_pad, %c0 : !dynamic_tensor_type
  %rC = tensor.dim %input_pad, %c1 : !dynamic_tensor_type
  %rH = arith.constant {{H_out}} : index
  %rW = arith.constant {{W_out}} : index
  %result_empty = tensor.empty(%rN, %rC, %rH, %rW) : !out_tensor_type
  %result_fill = linalg.fill ins(%zero: !dtype) outs(%result_empty: !out_tensor_type) -> !out_tensor_type
  %result = linalg.pooling_nchw_sum {dilations = dense<[{{dilations_H}}, {{dilations_W}}]> : tensor<2xi64>, strides = dense<[{{strides_H}}, {{strides_W}}]> : tensor<2xi64>} ins(%input_pad, %weights: !dynamic_tensor_type, !weights_tensor_type) outs(%result_fill: !out_tensor_type) -> !out_tensor_type
  util.return %result : !out_tensor_type
}

}
