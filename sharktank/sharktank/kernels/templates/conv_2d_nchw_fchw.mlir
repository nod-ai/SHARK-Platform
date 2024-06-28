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

util.func private @sharktank_conv_2d_nchw_fchw_{{strides_H}}_{{strides_W}}_{{dilations_H}}_{{dilations_W}}_{{dtype}} (
    %input_pad: !dynamic_tensor_type, %weights: !dynamic_tensor_type, %bias: tensor<?x!dtype>)
    -> !out_tensor_type {
  %zero = arith.constant 0: !dtype
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c2 = arith.constant 2: index
  %c3 = arith.constant 3: index

  %rN = tensor.dim %input_pad, %c0 : !dynamic_tensor_type
  %rC = tensor.dim %weights, %c0 : !dynamic_tensor_type
  %rH = arith.constant {{H_out}} : index
  %rW = arith.constant {{W_out}} : index
  %result_empty = tensor.empty(%rN, %rC, %rH, %rW) : !out_tensor_type
  %result_fill = linalg.fill ins(%zero: !dtype) outs(%result_empty: !out_tensor_type) -> !out_tensor_type
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<[{{dilations_H}}, {{dilations_W}}]> : tensor<2xi64>, strides = dense<[{{strides_H}}, {{strides_W}}]> : tensor<2xi64>} ins(%input_pad, %weights: !dynamic_tensor_type, !dynamic_tensor_type) outs(%result_fill: !out_tensor_type) -> !out_tensor_type
  %result_biased = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%result, %bias : !dynamic_tensor_type, tensor<?x!dtype>) outs(%result : !out_tensor_type) {
    ^bb0(%in: !dtype, %in_1: !dtype, %out: !dtype):
      %add = arith.addi %in, %in_1 : !dtype
      linalg.yield %add : !dtype
    } -> !out_tensor_type
  util.return %result_biased : !out_tensor_type
}

}
