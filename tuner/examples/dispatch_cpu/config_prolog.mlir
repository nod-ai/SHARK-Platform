module attributes { transform.with_named_sequence } {
  //===----------------------------------------------------------------------===//
  // Matmul tuning for linalg.mmt4d on CPU
  //===----------------------------------------------------------------------===//

  transform.named_sequence @match_mmt4d_f16_f16_f32(%root: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    // Match the linalg.mmt4d operation
    %op = transform.match_op %root { op_name = "linalg.mmt4d" } : (!transform.any_op) -> !transform.any_op

    // Optionally, include type constraints
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %op {
      ^bb0(%lhs: tensor<?x?x?x?xf16>, %rhs: tensor<?x?x?x?xf16>, %out: tensor<?x?x?x?xf32>):
        %mmt4d = linalg.mmt4d
          ins(%lhs, %rhs : tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>)
          outs(%out : tensor<?x?x?x?xf32>)
          -> tensor<?x?x?x?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)

    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}, %config: !transform.any_param {transform.readonly}) {
    // **Use transform.iree.set_compilation_info instead of transform.annotate**
    transform.iree.set_compilation_info %op %config : !transform.any_op, !transform.any_param

    transform.yield
  }
}
