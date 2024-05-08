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
        # TODO: This should iterate over a list of targets that are eligible so that
        # we can call in sequence and handle if the higher salience ones
        # return NotImplemented.
        found_target = self._target_cache.get(type_spec)
        if found_target is None:
            # Slow-path try to find it.
            found_target = self._match_target(type_spec)
            if found_target is None:
                raise NotImplementedError(
                    f"Overridable operator {self.__module__}.{self.__qualname__} does not "
                    f"have an implementation for argument types: "
                    f"{list(zip(tensor_names, type_spec))}"
                )
            self._target_cache[type_spec] = found_target
        return found_target(*bound_args.args, **bound_args.kwargs)

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

    def _match_target(self, type_spec: tuple):
        # TODO: This should return a list of targets that are eligible so that
        # we can call in sequence and handle if the higher salience ones
        # return NotImplemented.
        for override in reversed(self._overrides):
            for expected, actual in zip(override.type_spec, type_spec):
                if expected is None:
                    continue
                if issubclass(actual, expected):
                    continue
                break
            else:
                return override.target
        return None


def overridable(f):
    """Decorator to apply to overridable ops.

    Such ops can then have specializations stacked against them with the
    @override decorator.
    """
    dispatcher = SignatureDispatcher(f)
    functools.update_wrapper(dispatcher, f)
    return dispatcher
