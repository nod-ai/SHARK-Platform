
//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

  transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %variant_op
        , @match_op -> @apply_op_config
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} ////  module
