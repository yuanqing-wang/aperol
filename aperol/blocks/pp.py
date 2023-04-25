"""Momentum modules. """
import torch
from ..module import Block, Linear
from ..constants import MAX_IN, MAX_OUT

__all__ = ["MomentumUpdate"]

class MomentumUpdate(Block):
    def __init__(self):
        super().__init__()
        self.linear = Linear(
            activation=None, bias=False, max_in=MAX_IN-1, max_out=MAX_OUT-1,
        )

    def forward(self, v, e, x, p, config=None):
        """
        Examples
        --------
        >>> momentum_update = MomentumUpdate()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(2, 2, 4)
        >>> x = torch.zeros(2, 3, 6)
        >>> p = torch.zeros(2, 3, 7)
        >>> config = momentum_update.Config(10)
        >>> v1, e1, x1, p1 = momentum_update(v, e, x, p, config=config)
        >>> list(p1.shape)
        [2, 3, 10]
        >>> assert torch.isclose(v1, v).all()
        >>> assert torch.isclose(e1, e).all()
        """
        if config is None:
            config = self.sample()
        p = self.linear(p, config=config)
        return v, e, x, p