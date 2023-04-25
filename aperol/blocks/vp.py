"""Node to geometry modules. """

import torch
from ..module import Block, Linear
from ..constants import MAX_OUT

__all__ = ["Damping"]

class Damping(Block):
    """Damp geometry based on node embedding. """
    def __init__(self):
        super().__init__()
        self.linear_p = Linear(bias=False, activation=None, max_out=MAX_OUT-1)
        self.linear_v = Linear(
            bias=False, activation=torch.nn.Softplus(),
            max_out=MAX_OUT - 1,
        )

    def sample(self):
        return self.linear_p.sample()._replace(cls=self.__class__)

    def forward(self, v, e, x, p, config=None):
        if config is None:
            config = self.sample()
        p = self.linear_p(p, config=config)
        coefficients = self.linear_v(v, config=config).unsqueeze(-2)
        p = coefficients * p
        return v, e, x, p
