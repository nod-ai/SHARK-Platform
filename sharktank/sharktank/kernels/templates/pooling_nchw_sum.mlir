// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!dtype = {{dtype}}
!input_tensor_type = {{input_tensor_type}}
!weights_tensor_type = tensor<{{weights_H}}x{{weights_W}}x!dtype>
!padding_out_tensor_type = tensor<{{input_padding_N}}x{{input_padding_C}}x{{input_padding_H}}x{{input_padding_W}}x!dtype>
!out_tensor_type = {{pooling_nchw_sum_output_shape}}

module {

util.func private @sharktank_pooling_nchw_sum_{{input_dim_sizes}}_{{weights_H}}_{{weights_W}}_{{strides_H}}_{{strides_W}}_{{padding_H}}_{{padding_W}}_{{dilations_H}}_{{dilations_W}}_{{dtype}} (
    %input: !input_tensor_type)
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
  } : !input_tensor_type to !padding_out_tensor_type

  %result_empty = tensor.empty() : !out_tensor_type
  %result_fill = linalg.fill ins(%zero: !dtype) outs(%result_empty: !out_tensor_type) -> !out_tensor_type
  %result = linalg.pooling_nchw_sum {dilations = dense<[{{dilations_H}}, {{dilations_W}}]> : tensor<2xi64>, strides = dense<[{{strides_H}}, {{strides_W}}]> : tensor<2xi64>} ins(%input_pad, %weights: !padding_out_tensor_type, !weights_tensor_type) outs(%result_fill: !out_tensor_type) -> !out_tensor_type
  util.return %result : !out_tensor_type
}

}
