// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!dtype = {{dtype}}
!dynamic_tensor_type = tensor<?x?x?x?x!dtype>
!out_tensor_type = !dynamic_tensor_type

module {

util.func private @sharktank_conv_2d_nchw_fchw_{{strides_str}}_{{padding_str}}_{{dilations_str}}_{{dtype}} (
    %input: !dynamic_tensor_type, %weights: !dynamic_tensor_type)
    -> !out_tensor_type {
  %zero = arith.constant 0: !dtype
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c2 = arith.constant 2: index
  %c3 = arith.constant 3: index
  
  %input_pad = tensor.pad %input low[0, 0, {{padding}}] high[0, 0, {{padding}}] {
  ^bb0(%arg0 : index, %arg1 : index, %arg2: index, %arg3: index):
    tensor.yield %zero : !dtype
  } : !dynamic_tensor_type to !dynamic_tensor_type
  
  %pn = tensor.dim %input_pad,  %c0 : !dynamic_tensor_type
  %pc = tensor.dim %input_pad,  %c1 : !dynamic_tensor_type
  %ph = tensor.dim %input_pad,  %c2 : !dynamic_tensor_type
  %pw = tensor.dim %input_pad,  %c3 : !dynamic_tensor_type
  
  %result_empty = tensor.empty(%pn, %pc, %ph, %pw) : !out_tensor_type
  %result_fill = linalg.fill ins(%zero: !dtype) outs(%result_empty: !out_tensor_type) -> !out_tensor_type
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<[{{dilations}}]> : tensor<2xi64>, strides = dense<[{{strides}}]> : tensor<2xi64>} ins(%input_pad, %weights: !dynamic_tensor_type, !dynamic_tensor_type) outs(%result_fill: !out_tensor_type) -> !out_tensor_type
  util.return %result : !out_tensor_type
}

}
