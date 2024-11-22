import torch
import torch.nn.functional as F

from .. import ops

from .base import Theta, ThetaLayer
from .linear import LinearLayer


class ModulationOut:
    def __init__(self, shift, scale, gate):
        self.shift = shift
        self.scale = scale
        self.gate = gate


class ModulationLayer(ThetaLayer):
    def __init__(self, theta: Theta, double: bool):
        super().__init__(theta)

        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = LinearLayer(theta("lin"))

    def forward(self, vec: torch.Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        silu_result = ops.elementwise(F.silu, vec)
        out = self.lin(silu_result)[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )
