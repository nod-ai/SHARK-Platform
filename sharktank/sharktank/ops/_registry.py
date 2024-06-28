# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Signatures for dynamic dispatch of ops covering our fundamental tensor types."""

from typing import Any, Callable, Iterable, Optional, Union

import collections
import inspect
import functools

import torch
from torch import Tensor
from ..types import PrimitiveTensor, QuantizedTensor

__all__ = [
    "SignatureDispatcher",
    "overridable",
    "unbox_tensor",
]

_TargetOverride = collections.namedtuple(
    "_TargetOverride",
    "salience, target, type_spec, auto_unbox, auto_dequant",
)


# When an op is dispatched, it will be stashed here for testing to verify.
# Use _test_enable_last_op_dispatch(True) / _test_enable_last_op_dispatch(False)
# in test cases to enable/disable tracking of the last op dispatched.
# The last op can be queried with _test_get_last_op_dispatch().
_ENABLE_TEST_LAST_OP_DISPATCH = False
_TEST_LAST_OP_DISPATCH = None


def _test_enable_last_op_dispatch(en: bool = True):
    global _TEST_LAST_OP_DISPATCH
    global _ENABLE_TEST_LAST_OP_DISPATCH
    _TEST_LAST_OP_DISPATCH = None
    _ENABLE_TEST_LAST_OP_DISPATCH = en


def _test_get_last_op_dispatch():
    assert (
        _ENABLE_TEST_LAST_OP_DISPATCH
    ), "Cannot get last op dispatched without calling _test_enable_last_op_dispatch()"
    return _TEST_LAST_OP_DISPATCH


class SignatureDispatcher:
    """Replaces an overridable function with a tensor type base dispatcher.

    When overrides are registered, the computed target cache is cleared but
    between registrations, it is maintained for quick lookup by a tuple of
    tensor types in the order of the formal tensor arguments of the original
    function signature.
    """

    __slot__ = [
        # "_sig",
        # "_tensor_names",
        "_overrides",
        "_target_cache",
        "_trampoline",
    ]

    def __init__(self, sigf: Callable):
        self._target_cache = dict()
        self._trampoline: Optional[Callable] = None
        self._overrides: list[_TargetOverride] = []

    def __call__(self, *args, **kwargs):
        trampoline = self._trampoline
        assert trampoline is not None
        selected_override, *results = trampoline(self, *args, **kwargs)
        if _ENABLE_TEST_LAST_OP_DISPATCH:
            global _TEST_LAST_OP_DISPATCH
            _TEST_LAST_OP_DISPATCH = selected_override
        arity = len(results)
        if arity == 1:
            return results[0]
        elif arity == 0:
            return None
        else:
            return results

    def override(
        self,
        *type_spec: tuple[type, ...],
        salience: int = 0,
        auto_unbox: bool = True,
        auto_dequant: bool = False,
    ):
        def decorator(f):
            if f.__name__ == "_":
                f.__name__ = f"{self.__name__}__override"
            self._overrides.append(
                _TargetOverride(
                    salience=salience,
                    target=f,
                    type_spec=type_spec,
                    auto_unbox=auto_unbox,
                    auto_dequant=auto_dequant,
                )
            )
            self._overrides.sort(key=lambda v: v.salience)
            self._target_cache.clear()  # Need to recompute all targets
            return f

        return decorator

    def find_overrides(self, tensors: tuple[Any, ...]) -> Iterable[Callable]:
        """Finds the most salient override for the given named tensors."""
        type_spec = tuple(type(t) for t in tensors)
        found_targets = self._target_cache.get(type_spec)
        if found_targets is None:
            # Slow-path try to find it.
            found_targets = self._match_targets(type_spec)
            self._target_cache[type_spec] = found_targets
        return reversed(found_targets)

    def fail(self, tensors: tuple[Any, ...]):
        spec = [type(t) for t in tensors]
        raise NotImplementedError(
            f"Overridable operator {self.__module__}.{self.__qualname__} does not "
            f"have an implementation for argument types: "
            f"{spec}"
        )

    def trampoline(self, trampoline: Callable):
        assert self._trampoline is None
        self._trampoline = trampoline

    def _match_targets(self, type_spec: tuple):
        targets = []
        for override in self._overrides:
            override_type_spec = override.type_spec
            if len(override_type_spec) != len(type_spec):
                continue
            for expected, actual in zip(override.type_spec, type_spec):
                if expected is None:
                    continue
                if issubclass(actual, expected):
                    continue
                # We expect kernels which are parameterized on Tensor to
                # unbox things that are isomorphic to it.
                is_expected_tensor = issubclass(expected, Tensor)
                if is_expected_tensor:
                    if override.auto_unbox and issubclass(actual, PrimitiveTensor):
                        continue
                    # Similarly, we conditionally allow auto dequant.
                    if override.auto_dequant and issubclass(actual, QuantizedTensor):
                        continue
                break
            else:
                targets.append(override.target)
        return targets


def overridable(f):
    """Decorator to apply to overridable ops.

    Such ops can then have specializations stacked against them with the
    @override decorator.
    """
    dispatcher = SignatureDispatcher(f)
    functools.update_wrapper(dispatcher, f)
    return dispatcher


def unbox_tensor(t: Any) -> Tensor:
    """Unboxes a value that can be isomorphically interpreted as a Tensor."""
    if isinstance(t, Tensor):
        return t
    elif isinstance(t, PrimitiveTensor):
        return t.as_torch()
    elif isinstance(t, QuantizedTensor):
        return t.unpack().dequant()
    raise ValueError(f"Expected a Tensor or PrimitiveTensor but got {type(t)}")
