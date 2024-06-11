#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> ()>

#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> ()>

module {

util.func private @mmt_scaled_q8(%arg0 : tensor<{{m}}x{{k}}x{{lowp_type}}>, %arg1 : tensor<{{n}}x{{k}}x{{lowp_type}}>, %arg2 : tensor<{{lowp_type}}>, %arg3 : tensor<{{lowp_type}}>) -> tensor<{{m}}x{{n}}x{{a_type}}> {
    %zerof32 = arith.constant 0.0 : {{a_type}}
    %emptyf32 = tensor.empty() : tensor<{{m}}x{{n}}x{{a_type}}>
    %fullf32 = linalg.fill ins (%zerof32 : {{a_type}}) outs (%emptyf32 : tensor<{{m}}x{{n}}x{{a_type}}>)  -> tensor<{{m}}x{{n}}x{{a_type}}>

%mmt = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
ins(%arg0, %arg1 : tensor<{{m}}x{{k}}x{{lowp_type}}>, tensor<{{n}}x{{k}}x{{lowp_type}}>) outs(%fullf32 : tensor<{{m}}x{{n}}x{{a_type}}>) {
^bb0(%lhs: {{lowp_type}}, %rhs: {{lowp_type}}, %out0: {{a_type}}):
%lhs_i32 = arith.extsi %lhs : {{lowp_type}} to i32
%lhs_f32 = arith.sitofp %lhs_i32 : i32 to {{a_type}}
%rhs_i32 = arith.extsi %rhs : {{lowp_type}} to i32
%rhs_f32 = arith.sitofp %rhs_i32 : i32 to {{a_type}}

%mul = arith.mulf %lhs_f32, %rhs_f32 : {{a_type}}
%add = arith.addf %out0, %mul : {{a_type}}
linalg.yield %add : {{a_type}}
} -> tensor<{{m}}x{{n}}x{{a_type}}>


%empty = tensor.empty() : tensor<{{m}}x{{n}}x{{a_type}}>
    %result = linalg.generic {indexing_maps = [#map4, #map5, #map5, #map4], iterator_types = ["parallel", "parallel"]}
ins(%mmt, %arg2, %arg3 : tensor<{{m}}x{{n}}x{{a_type}}>, tensor<{{lowp_type}}>, tensor<{{lowp_type}}>) outs(%empty : tensor<{{m}}x{{n}}x{{a_type}}>) {
^bb0(%in: {{a_type}}, %scale0 : {{lowp_type}}, %scale1 : {{lowp_type}}, %out: {{a_type}}):
%scale0_i32 = arith.extsi %scale0 : {{lowp_type}} to i32
%scale0_f32 = arith.sitofp %scale0_i32 : i32 to {{a_type}}
%scale1_i32 = arith.extsi %scale1 : {{lowp_type}} to i32
%scale1_f32 = arith.sitofp %scale1_i32 : i32 to {{a_type}}

%scale_f32 = arith.mulf %scale0_f32, %scale1_f32 : {{a_type}}
%scaled = arith.mulf %scale_f32, %in : {{a_type}}

linalg.yield %scaled : {{a_type}}
} -> tensor<{{m}}x{{n}}x{{a_type}}>

util.return %result : tensor<{{m}}x{{n}}x{{a_type}}>
}

}
