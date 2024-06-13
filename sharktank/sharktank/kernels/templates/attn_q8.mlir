#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> ()>
#map7 = affine_map<(d0, d1, d2) -> (d0)>
#map8 = affine_map<(d0, d1, d2) -> (d1)>

#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
#map6 = affine_map<(d0, d1) -> (d1)>

module {

util.func private @attn_q8(%query : tensor<{{m}}x{{k}}x{{lowp_type}}>, %key : tensor<{{n}}x{{k}}x{{lowp_type}}>, %value : tensor<{{p}}x{{n}}x{{lowp_type}}>, %query_s : tensor<{{m}}x{{lowp_type}}>, %key_s : tensor<{{n}}x{{lowp_type}}>, %value_s : tensor<{{p}}x{{lowp_type}}>, %query_zp: tensor<{{m}}x{{lowp_type}}>, %key_zp: tensor<{{n}}x{{lowp_type}}>, %value_zp : tensor<{{p}}x{{lowp_type}}>, %attn_mask : tensor<{{m}}x{{n}}xi1>, %randoms : tensor<{{m}}x{{n}}x{{a_type}}>, %dropout_p_t : tensor<f32>, %is_causal_t : tensor<i1>, %scale_t : tensor<i1>) -> tensor<{{m}}x{{p}}x{{a_type}}> {

iree_input.tensor.trace "FOOBAR" = [%query : tensor<{{m}}x{{k}}x{{lowp_type}}>, %key : tensor<{{n}}x{{k}}x{{lowp_type}}>, %value : tensor<{{p}}x{{n}}x{{lowp_type}}>, %query_s : tensor<{{m}}x{{lowp_type}}>, %key_s : tensor<{{n}}x{{lowp_type}}>, %value_s : tensor<{{p}}x{{lowp_type}}>, %query_zp: tensor<{{m}}x{{lowp_type}}>, %key_zp: tensor<{{n}}x{{lowp_type}}>, %value_zp : tensor<{{p}}x{{lowp_type}}>, %attn_mask : tensor<{{m}}x{{n}}xi1>, %randoms : tensor<{{m}}x{{n}}x{{a_type}}>, %dropout_p_t : tensor<f32>, %is_causal_t : tensor<i1>, %scale_t : tensor<i1>]

%zerof32 = arith.constant 0.0 : {{a_type}}
%onef32 = arith.constant 1.0 : {{a_type}}
%inff32 = arith.constant 0xFF800000 : {{a_type}} // negative infinity
%k = arith.constant {{k}}.0 : {{a_type}}

%dropout_p = tensor.extract %dropout_p_t[] : tensor<f32>
%is_causal = tensor.extract %is_causal_t[] : tensor<i1>
%scale = tensor.extract %scale_t[] : tensor<i1>

// scale
%sdpa_scale_value = math.rsqrt %k : {{a_type}}
%sdpa_scale = arith.select %scale, %sdpa_scale_value, %onef32 : {{a_type}}

// bias
%bias_empty = tensor.empty() : tensor<{{m}}x{{n}}x{{a_type}}>
%bias_init = linalg.fill ins (%zerof32 : {{a_type}}) outs (%bias_empty : tensor<{{m}}x{{n}}x{{a_type}}>) -> tensor<{{m}}x{{n}}x{{a_type}}>

// if is_causal
%mask_partial_empty = tensor.empty(): tensor<{{m}}x{{n}}x{{a_type}}>
%mask_partial_init = linalg.fill ins (%onef32 : {{a_type}}) outs (%bias_empty : tensor<{{m}}x{{n}}x{{a_type}}>) -> tensor<{{m}}x{{n}}x{{a_type}}>
//%mask_partial = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel"]}
//outs(%mask_partial_init: tensor<{{m}}x{{n}}x{{a_type}}>) {
//^bb0(%out : {{a_type}}):
//  %i = linalg.index 0 : index
//  %j = linalg.index 1 : index
//  %not_tril = index.cmp sgt(%i, %j)
//  %select = arith.select %not_tril, %onef32, %inff32 : {{a_type}}
//  linalg.yield %select : {{a_type}}
//} -> tensor<{{m}}x{{n}}x{{a_type}}>

// attn_mask

%mask = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]}
ins(%attn_mask : tensor<{{m}}x{{n}}xi1>) outs(%bias_init : tensor<{{m}}x{{n}}x{{a_type}}>) {
^bb0(%m : i1, %out : f32):
  %select = arith.select %m, %zerof32, %inff32 : {{a_type}}
  %add = arith.addf %out, %select : {{a_type}}
  linalg.yield %add : {{a_type}}
} -> tensor<{{m}}x{{n}}x{{a_type}}>

// matmul
%empty_mmt = tensor.empty() : tensor<{{m}}x{{n}}x{{a_type}}>
%full_mmt = linalg.fill ins (%zerof32 : {{a_type}}) outs (%empty_mmt : tensor<{{m}}x{{n}}x{{a_type}}>)  -> tensor<{{m}}x{{n}}x{{a_type}}>

%mmt = linalg.generic {indexing_maps = [#map0, #map1, #map7, #map8, #map7, #map8, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
ins(%query, %key, %query_s, %key_s, %query_zp, %key_zp : tensor<{{m}}x{{k}}x{{lowp_type}}>, tensor<{{n}}x{{k}}x{{lowp_type}}>, tensor<{{m}}x{{lowp_type}}>, tensor<{{n}}x{{lowp_type}}>, tensor<{{m}}x{{lowp_type}}>, tensor<{{n}}x{{lowp_type}}>) outs(%full_mmt : tensor<{{m}}x{{n}}x{{a_type}}>) {
^bb0(%lhs: {{lowp_type}}, %rhs: {{lowp_type}}, %scale0: {{lowp_type}}, %scale1: {{lowp_type}}, %zp0: {{lowp_type}}, %zp1: {{lowp_type}}, %out0: {{a_type}}):
  %lhs_i32 = arith.extsi %lhs : {{lowp_type}} to i32
  %lhs_f32 = arith.sitofp %lhs_i32 : i32 to {{a_type}}
  %rhs_i32 = arith.extsi %rhs : {{lowp_type}} to i32
  %rhs_f32 = arith.sitofp %rhs_i32 : i32 to {{a_type}}
  %zp0_i32 = arith.extsi %zp0 : {{lowp_type}} to i32
  %zp0_f32 = arith.sitofp %zp0_i32 : i32 to {{a_type}}
  %zp1_i32 = arith.extsi %zp1 : {{lowp_type}} to i32
  %zp1_f32 = arith.sitofp %zp1_i32 : i32 to {{a_type}}

  %lhs_offset = arith.subf %lhs_f32, %zp0_f32 : {{a_type}}
  %rhs_offset = arith.subf %rhs_f32, %zp1_f32 : {{a_type}}

  %scale0_i32 = arith.extsi %scale0 : {{lowp_type}} to i32
  %scale0_f32 = arith.sitofp %scale0_i32 : i32 to {{a_type}}
  %scale1_i32 = arith.extsi %scale1 : {{lowp_type}} to i32
  %scale1_f32 = arith.sitofp %scale1_i32 : i32 to {{a_type}}

  %scaled_lhs = arith.mulf %scale0_f32, %lhs_offset : {{a_type}}
  %scaled_rhs = arith.mulf %scale1_f32, %rhs_offset : {{a_type}}
  %mul = arith.mulf %scaled_lhs, %scaled_rhs : {{a_type}}
  %add = arith.addf %out0, %mul : {{a_type}}
  linalg.yield %add : {{a_type}}
} -> tensor<{{m}}x{{n}}x{{a_type}}>

// bias
%biased_empty = tensor.empty() : tensor<{{m}}x{{n}}x{{a_type}}>
%biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%mmt, %mask : tensor<{{m}}x{{n}}x{{a_type}}>, tensor<{{m}}x{{n}}x{{a_type}}>)
    outs(%biased_empty : tensor<{{m}}x{{n}}x{{a_type}}>) -> tensor<{{m}}x{{n}}x{{a_type}}>

// softmax
%softmax_empty = tensor.empty() : tensor<{{m}}x{{n}}x{{a_type}}>
%softmaxed = linalg.softmax dimension(1) ins(%biased : tensor<{{m}}x{{n}}x{{a_type}}>) outs(%softmax_empty : tensor<{{m}}x{{n}}x{{a_type}}>) -> tensor<{{m}}x{{n}}x{{a_type}}>

// dropout
%dropout_empty = tensor.empty() : tensor<{{m}}x{{n}}x{{a_type}}>
%dropouted  = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]}
ins(%softmaxed, %randoms : tensor<{{m}}x{{n}}x{{a_type}}>, tensor<{{m}}x{{n}}x{{a_type}}>) outs(%dropout_empty : tensor<{{m}}x{{n}}x{{a_type}}>) {
^bb0(%s : {{a_type}}, %r : {{a_type}}, %out : f32):
  %rand = arith.cmpf olt, %r, %dropout_p : f32
  %select = arith.select %rand, %zerof32, %s : {{a_type}}
  linalg.yield %select : {{a_type}}
} -> tensor<{{m}}x{{n}}x{{a_type}}>

iree_input.tensor.trace  "FOOBAR" = [%dropouted: tensor<{{m}}x{{n}}x{{a_type}}>]

// matmul
%empty_mmt2 = tensor.empty() : tensor<{{m}}x{{p}}x{{a_type}}>
%full_mmt2 = linalg.fill ins (%zerof32 : {{a_type}}) outs (%empty_mmt2 : tensor<{{m}}x{{p}}x{{a_type}}>)  -> tensor<{{m}}x{{p}}x{{a_type}}>
%mmt2 = linalg.generic {indexing_maps = [#map0, #map1, #map8, #map8, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
ins(%dropouted, %value, %value_s, %value_zp : tensor<{{m}}x{{n}}x{{a_type}}>, tensor<{{p}}x{{n}}x{{lowp_type}}>, tensor<{{p}}x{{lowp_type}}>, tensor<{{p}}x{{lowp_type}}>) outs(%full_mmt2 : tensor<{{m}}x{{p}}x{{a_type}}>) {
^bb0(%lhs: {{a_type}}, %rhs: {{lowp_type}}, %scale1: {{lowp_type}}, %zp: {{lowp_type}}, %out0: {{a_type}}):
  %rhs_i32 = arith.extsi %rhs : {{lowp_type}} to i32
  %rhs_f32 = arith.sitofp %rhs_i32 : i32 to {{a_type}}
  %zp_i32 = arith.extsi %zp : {{lowp_type}} to i32
  %zp_f32 = arith.sitofp %zp_i32 : i32 to {{a_type}}

  %rhs_offset = arith.subf %rhs_f32, %zp_f32 : {{a_type}}

  %scale1_i32 = arith.extsi %scale1 : {{lowp_type}} to i32
  %scale1_f32 = arith.sitofp %scale1_i32 : i32 to {{a_type}}

  %scaled_rhs = arith.mulf %scale1_f32, %rhs_offset : {{a_type}}
  %mul = arith.mulf %lhs, %scaled_rhs : {{a_type}}
  %add = arith.addf %out0, %mul : {{a_type}}
  linalg.yield %add : {{a_type}}
} -> tensor<{{m}}x{{p}}x{{a_type}}>

iree_input.tensor.trace  "FOOBAR" = [%mmt2: tensor<{{m}}x{{p}}x{{a_type}}>]

util.return %mmt2 : tensor<{{m}}x{{p}}x{{a_type}}>
}

}
