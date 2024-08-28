# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from .. import ops
from .base import Theta, ThetaLayer
from safetensors.torch import save_file


class RMSNormLayer(ThetaLayer):
    """Computes the unbiased full RMS layer normalization.

    Because normalization is sensitive to floating point error, we support
    an explicit dtype that the input will be casted to prior to performing
    the compute. The result will be cast back to the input dtype.
    """

    def __init__(
        self,
        theta: Theta,
        *,
        weight_name: str = "weight",
        epsilon: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        debug_save_file=None,
    ):
        super().__init__(theta)
        self.weight = self.theta_tensor(weight_name)
        self.epsilon = epsilon
        self.dtype = dtype
        self.debug_save_file = debug_save_file

    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        print("norm dtype: ", self.dtype)
        print("orgi_dtype: ", orig_dtype)

        x = x.to(self.dtype)
        norm = ops.rms_norm(x, self.weight, epsilon=self.epsilon)
        # Will automatically upcast to the dtype of the weight, which is
        # often in higher precision. Downcast back to expected.
        norm = norm.to(orig_dtype)
        if self.debug_save_file is not None:
            save_file(
                {
                    "input": x,
                    "variance": torch.tensor(self.epsilon),
                    "weight": self.weight.as_torch(),
                    "output": norm,
                },
                self.debug_save_file,
            )
        return norm
