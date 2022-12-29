"""Node to geometry modules. """

import torch
from ..module import BlockModule, Linear
from ..constants import MAX_OUT

class Damping(BlockModule):
    """Damp geometry based on node embedding. """
    def __init__(self):
        super().__init__()
        self.linear_x = Linear(bias=False, activation=None, max_out=MAX_OUT-1)
        self.linear_v = Linear(
            bias=False, activation=torch.nn.SoftPlus(),
            max_out=MAX_OUT - 1,
        )

    def sample(self):
        return self.linear_x.sample()._replace(cls=self.__class__)

    def forward(self, v, e, x, config=None):
        if config is None:
            config = self.sample()
        x_new = self.linear_x(x[..., 1:], config=config)
        coefficients = self.linear_v(v, config=config).unsqueeze(-2)
        x_new = coefficients * x_new
        x = torch.cat([x[..., :1], x_new], dim=-1)
        return v, e, x
