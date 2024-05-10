# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Signatures for dynamic dispatch of ops covering our fundamental tensor types."""

from typing import Callable, Union

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
        "_sig",
        "_tensor_names",
        "_target_cache",
    ]

    def __init__(self, sigf: Callable):
        self._sig = inspect.signature(sigf)
        self._target_cache = dict()
        tensor_names: set[str] = set()
        self._tensor_names = tensor_names
        for p in self._sig.parameters.values():
            annot = p.annotation
            if annot is AnyTensor:
                tensor_names.add(p.name)
        self._overrides: list[_TargetOverride] = []

    def __call__(self, *args, **kwargs):
        tensor_names = self._tensor_names
        bound_args = self._sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        value_spec = tuple(bound_args.arguments[k] for k in tensor_names)
        type_spec = tuple(type(v) for v in value_spec)
        found_targets = self._target_cache.get(type_spec)
        if found_targets is None:
            # Slow-path try to find it.
            found_targets = self._match_targets(type_spec)
            self._target_cache[type_spec] = found_targets
        global _TEST_LAST_OP_DISPATCH
        for found_target in reversed(found_targets):
            _TEST_LAST_OP_DISPATCH = found_target
            result = found_target(*bound_args.args, **bound_args.kwargs)
            if result is not NotImplemented:
                return result
        raise NotImplementedError(
            f"Overridable operator {self.__module__}.{self.__qualname__} does not "
            f"have an implementation for argument types: "
            f"{list(zip(tensor_names, type_spec))}"
        )

    def override(self, *, salience: int = 0, **kwargs: type):
        tensor_names = self._tensor_names
        type_spec = tuple(kwargs.get(n) for n in tensor_names)
        if len(kwargs) > len(tensor_names):
            raise TypeError(
                f"Extra tensor types in override: {kwargs.keys() - set(tensor_names)}"
            )

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
