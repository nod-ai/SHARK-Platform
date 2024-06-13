# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict

import torch
import torch.nn as nn

from ..types import (
    InferenceTensor,
    Theta,
)
from ..utils import debugging

__all__ = [
    "LinearLayer",
    "RotaryEmbeddingLayer",
    "RMSNormLayer",
    "ThetaLayer",
    "TokenEmbedding",
]


class BaseLayer(nn.Module):
    """Base class of all of our layers."""

    def trace_tensor(
        self, key: str, t: torch.Tensor, *, values: bool = True, golden: bool = False
    ):
        debugging.trace_tensor(key, t, values=values, golden=golden)

    def trace_tensors(
        self,
        key: str,
        tensors: Dict[str, torch.Tensor],
        *,
        values: bool = True,
        golden: bool = False,
    ):
        debugging.trace_tensors(key, tensors, values=values, golden=golden)

    def trace_golden(self, key: str, t: torch.Tensor):
        debugging.trace_tensor(key, t, golden=True)

    def trace_goldens(self, key: str, tensors: Dict[str, torch.Tensor]):
        debugging.trace_tensors(key, tensors, golden=True)

    def assert_not_nan(self, *ts: torch.Tensor):
        """Checks whether tensors have nan values in them.

        Must be enabled via a global switch as this kind of checking is not
        accelerator or compilation friendly.
        """
        if debugging.flags.enable_nan_checks:
            for t in ts:
                if torch.isnan(t).any():
                    raise AssertionError(f"Tensor contains nans! {t}")


class ThetaLayer(BaseLayer):
    "Base class for layers that derive parameters from a Theta object."

    def __init__(self, theta: Theta):
        super().__init__()
        self.theta = theta

    def theta_tensor(self, name: str) -> InferenceTensor:
        # TODO: We may need to do some bookkeeping here to ensure export
        # tracks all of these.
        return self.theta.tensor(name)
