// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!q_type = tensor<?x{{l}}x{{d}}x{{i_type}}>
!k_type = tensor<?x{{s}}x{{d}}x{{i_type}}>
!v_type = tensor<?x{{s}}x{{e}}x{{i_type}}>
!o_type = tensor<?x{{l}}x{{e}}x{{o_type}}>
!o_dyn_type = tensor<?x?x?x{{o_type}}>

module {

util.func private @sharktank_flash_attention_{{l}}_{{s}}_{{d}}_{{e}}_{{i_type}}_{{o_type}}(
    %q: !q_type,
    %k: !k_type,
    %v: !v_type) -> !o_type {
        
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index

        %b = tensor.dim %q, %c0 : !q_type
        %l = tensor.dim %q, %c1 : !q_type
        %d = tensor.dim %q, %c2 : !q_type
        %e = tensor.dim %v, %c2 : !q_type

        %di = arith.index_cast %d : index to i32
        %df = arith.sitofp %di : i32 to f32
        %scale = math.sqrt %df : f32

        %empty_dyn = tensor.empty(%b, %l, %e) : !o_dyn_type
        %empty = tensor.cast %empty_dyn : !o_dyn_type to !o_type

        %f0 = arith.constant 0.0 : {{o_type}}
        %fill = linalg.fill ins(%f0 : f32) outs(%empty : !o_type)  -> !o_type

        %atten = iree_linalg_ext.attention ins(%q, %k, %v, %scale : !q_type, !k_type, !v_type, f32) outs(%fill : !o_type) -> !o_type
        util.return %atten : !o_type
    }
}
