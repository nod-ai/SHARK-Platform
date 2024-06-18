// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!dtype = {{dtype}}
!a_tensor_type = tensor<?x{{m}}x{{k}}x!dtype>
!b_tensor_type = tensor<?x{{n}}x{{k}}x!dtype>
!c_tensor_type = tensor<?x{{m}}x{{n}}x!dtype>

module {

util.func private @sharktank_batch_matmul_transpose_b_{{m}}_{{n}}_{{k}}_{{dtype}}(
    %a: !a_tensor_type, %b: !b_tensor_type)
    -> !c_tensor_type {
  %zero = arith.constant 0: !dtype
  %c0 = arith.constant 0: index
  %batch = tensor.dim %a, %c0 : !a_tensor_type
  %result_empty = tensor.empty(%batch) : !c_tensor_type
  %result_fill = linalg.fill ins(%zero: !dtype) outs(%result_empty: !c_tensor_type) -> !c_tensor_type
  %result = linalg.batch_matmul_transpose_b ins(%a, %b: !a_tensor_type, !b_tensor_type) outs(%result_fill: !c_tensor_type) -> !c_tensor_type

  util.return %result : !c_tensor_type
}

}
