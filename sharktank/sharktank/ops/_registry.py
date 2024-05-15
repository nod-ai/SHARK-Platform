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
from ..types import InferenceTensor

__all__ = [
    "AnyTensor",
    "SignatureDispatcher",
    "overridable",
]

AnyTensor = Union[torch.Tensor, InferenceTensor]

_TargetOverride = collections.namedtuple(
    "_TargetOverride", "salience, target, type_spec"
)


# When an op is dispatched, it will be stashed here for testing to verify.
_TEST_LAST_OP_DISPATCH = None


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
        global _TEST_LAST_OP_DISPATCH
        _TEST_LAST_OP_DISPATCH = selected_override
        arity = len(results)
        if arity == 1:
            return results[0]
        elif arity == 0:
            return None
        else:
            return results

    def override(self, *type_spec: tuple[type, ...], salience: int = 0):
        def decorator(f):
            if f.__name__ == "_":
                f.__name__ = f"{self.__name__}__override"
            self._overrides.append(
                _TargetOverride(salience=salience, target=f, type_spec=type_spec)
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
            for expected, actual in zip(override.type_spec, type_spec):
                if expected is None:
                    continue
                if issubclass(actual, expected):
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
