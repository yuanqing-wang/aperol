"""Node to geometry modules. """

import torch
from ..module import Module, Linear
from ..constants import MAX_OUT

__all__ = ["Damping"]

class Damping(Module):
    """Damp geometry based on node embedding. """
    def __init__(self):
        super().__init__()
        self.linear_p = Linear(bias=False, activation=None, max_out=MAX_OUT-1)
        self.linear_v = Linear(
            bias=False, activation=torch.nn.Softplus(),
            max_out=MAX_OUT - 1,
        )

    def forward(self, v, e, x, p):
        p = self.linear_p(p)
        coefficients = self.linear_v(v).unsqueeze(-2)
        p = coefficients * p
        return v, e, x, p
