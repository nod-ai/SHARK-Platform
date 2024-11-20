# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This code implements matcher functions for MLIR modules using python bindings.

from abc import abstractmethod

from .common import *
from iree.compiler import ir  # type: ignore

class LinalgIteratorType(str, Enum):
    parallel = "#linalg.iterator_type<parallel>"
    reduction = "#linalg.iterator_type<reduction>"


def walk_collect_ops(
    op: ir.Operation,
    ops: list[ir.Operation],
    fn,
) -> ir.WalkResult:
    if fn(op):
        ops.append(op)
    return ir.WalkResult.ADVANCE


def get_ops(op: ir.Operation, fn):
    ops: list[ir.Operation] = []
    op.opview.walk(
        lambda op: walk_collect_ops(op, ops, fn),
        ir.WalkOrder.POST_ORDER,
    )
    return ops


def get_ops_from_module(module: ir.Module, fn):
    ops: list[ir.Operation] = []
    for op in module.body.operations:
        op.walk(
            lambda op: walk_collect_ops(op, ops, fn),
            ir.WalkOrder.POST_ORDER,
        )
    return ops


def get_named_ops(module: ir.Module, name: str):
    return get_ops_from_module(
        module,
        lambda op: op.name == name
    )


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
    m = ir.AffineDimExpr.get(0, context)
    n = ir.AffineDimExpr.get(1, context)
    k = ir.AffineDimExpr.get(2, context)
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


class NamedOpMatcher:
    def __init__(self, op_name):
        self.op_name = op_name

    @abstractmethod
    def match(self, op: ir.Operation) -> bool:
        return op.name == self.op_name

    def get_matched_ops(self, op: ir.Operation):
        return get_ops(
            op,
            lambda nestedOp: self.match(nestedOp)
        )


class GenericOpMatcher(NamedOpMatcher):
    def __init__(self):
        super().__init__("linalg.generic")

    @abstractmethod
    def match_operands(self, operands: ir.OpOperandList) -> bool:
        """Match the operands of the linalg op."""
        pass
    
    @abstractmethod
    def match_indexing_maps(self, maps: list[ir.AffineMap]) -> bool:
        """Match the indexing_maps of the linalg op."""
        pass
    
    @abstractmethod
    def match_iterator_types(self, iterators: list[LinalgIteratorType]) -> bool:
        """Match the iterator_types of the linalg op."""
        pass
    
    @abstractmethod
    def match_body(self, body_region: ir.Region) -> bool:
        """Match the body operations of the linalg op."""
        pass

    def match(self, op: ir.Operation) -> bool:
        if op.name != self.op_name:
            return False

        if not self.match_operands(op.operands):
            return False
        
        iterators_attr = None
        maps_attr = None
        for attr in op.opview.attributes:
            if attr.name == "iterator_types" and isinstance(attr.attr, ir.ArrayAttr):
                iterators_attr = attr.attr
            if attr.name == "indexing_maps" and isinstance(attr.attr, ir.ArrayAttr):
                maps_attr = attr.attr
        if maps_attr is None or iterators_attr is None:
            return False

        iterators: list[LinalgIteratorType] = []
        for iterator in iterators_attr:
            if str(iterator) == LinalgIteratorType.parallel:
                iterators.append(LinalgIteratorType.parallel)
            elif str(iterator) == LinalgIteratorType.reduction:
                iterators.append(LinalgIteratorType.reduction)
            else:
                return False
        if not self.match_iterator_types(iterators):
            return False

        maps: list[ir.AffineMap] = []
        for map in maps_attr:
            maps.append(map.value)
        if not self.match_indexing_maps(maps):
            return False

        if len(op.regions) != 1:
            return False
        if not self.match_body(op.regions[0]):
            return False
        return True


class MmtOpMatcher(GenericOpMatcher):
    def match_operands(self, operands: ir.OpOperandList) -> bool:
        if (len(operands) != 3):
            return False
        shaped_types: list[ir.ShapedType] = []
        for operand in operands:
            if not isinstance(operand.type, ir.ShapedType):
                return False
            shaped_types.append(operand.type)
        # Check LHS type
        if (shaped_types[0].rank != 2):
            return False
        # Check RHS type
        if (shaped_types[1].rank != 2):
            return False
        # Check ACC type
        if (shaped_types[2].rank != 2):
            return False
        return True
    
    def match_indexing_maps(self, maps: list[ir.AffineMap]) -> bool:
        if len(maps) != 3:
            return False
        return get_mmt_indexing_maps(maps[0].context) == maps
    
    def match_iterator_types(self, iterators: list[LinalgIteratorType]) -> bool:
        return True
    
    def match_body(self, body_region: ir.Region) -> bool:
        return True


class Conv2dNHWCOpMatcher(NamedOpMatcher):
    def __init__(self):
        super().__init__("linalg.conv_2d_nhwc_hwcf")
