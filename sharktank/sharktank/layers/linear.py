# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from .. import ops
from .base import Theta, ThetaLayer


class LinearLayer(ThetaLayer):
    """Linear layer which computes:

    ```
    matmul(x, weight.T)
    ```

    Whether the weight is transposed as part of the calculation can be
    controlled with `transpose_weight=` (default true).
    """

    def __init__(
        self,
        theta: Theta,
        *,
        weight_name: str = "weight",
        bias_name: str = "bias",
        transpose_weight: bool = True,
    ):
        super().__init__(theta)
        self.weight = self.theta_tensor(weight_name)
        self.bias = None
        if bias_name in self.theta.keys:
            self.bias = self.theta_tensor(bias_name)
        self.transpose_weight = transpose_weight

    def forward(self, x: torch.Tensor):
        x = ops.matmul(x, self.weight, transpose_rhs=self.transpose_weight)
        if self.bias is not None:
            x = ops.elementwise(torch.add, x, self.bias)
        return x
