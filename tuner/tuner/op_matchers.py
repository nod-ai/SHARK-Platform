# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This code implements matcher functions for MLIR modules using python bindings.

from .common import *
from iree.turbine.transforms.rewriter import *
from iree.turbine.transforms.builder import *

class LinalgIteratorType(str, Enum):
    parallel = "#linalg.iterator_type<parallel>"
    reduction = "#linalg.iterator_type<reduction>"


def has_iterator_types(attr: ir.ArrayAttr, iterators: list[LinalgIteratorType]):
    if len(attr) != len(iterators):
        return False
    for i, a in enumerate(attr):
        if str(a) != iterators[i]:
            return False
    return True


def has_indexing_maps(attr: ir.ArrayAttr, maps: list[ir.AffineMap]):
    if len(attr) != len(maps):
        return False
    map_attrs = []
    for map in maps:
        map_attrs.append(ir.AffineMapAttr.get(map))
    for i, map_attr in enumerate(attr):
        if map_attrs[i] != map_attr:
            return False
    return True


def get_mmt_iterator_types(context: ir.Context):
    return [
        LinalgIteratorType.parallel,
        LinalgIteratorType.parallel,
        LinalgIteratorType.reduction,
    ]


def has_mmt_iterator_types(attr: ir.ArrayAttr):
    mmt_iterators = get_mmt_iterator_types(attr.context)
    return has_iterator_types(attr, mmt_iterators)


def get_mmt_indexing_maps(context: ir.Context):
    m = ir.AffineDimExpr.get(0)
    n = ir.AffineDimExpr.get(1)
    k = ir.AffineDimExpr.get(2)
    lhs_exprs = [m, k]
    lhs_map = ir.AffineMap.get(3, 0, exprs=lhs_exprs, context=context)
    rhs_exprs = [n, k]
    rhs_map = ir.AffineMap.get(3, 0, exprs=rhs_exprs, context=context)
    acc_exprs = [m, n]
    acc_map = ir.AffineMap.get(3, 0, exprs=acc_exprs, context=context)
    return [lhs_map, rhs_map, acc_map]


def has_mmt_indexing_maps(attr: ir.ArrayAttr):
    if len(attr) != 3:
        return False
    mmt_maps = get_mmt_indexing_maps(attr.context)
    return has_indexing_maps(attr, mmt_maps)

class MmtMatcher(NamedOpMatcher):
    def __init__(self, builder: Builder):
        super().__init__("linalg.generic")
        self.builder = builder

    def match(self, op: ir.Operation):
        if (len(op.operands) != 3):
            return None
        lhs_dims = self.builder.get_tensor_dims(op.operands[0].type)
        if (len(lhs_dims) != 2):
            return None
        rhs_dims = self.builder.get_tensor_dims(op.operands[1].type)
        if (len(rhs_dims) != 2):
            return None
        acc_dims = self.builder.get_tensor_dims(op.operands[2].type)
        if (len(acc_dims) != 2):
            return None
        for attr in op.opview.attributes:
            if attr.name == "iterator_types" and not has_mmt_iterator_types(attr.attr):
                return None
            if (attr.name == "indexing_maps" and not has_mmt_indexing_maps(attr.attr)):
                return None
        # TODO(Max191): Check the op body
        return OpMatchResult(op)
