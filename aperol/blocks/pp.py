"""Momentum modules. """
import torch
from ..module import Module, Linear
from ..constants import MAX_IN, MAX_OUT

__all__ = ["MomentumUpdate"]

class MomentumUpdate(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(
            activation=None, bias=False, max_in=MAX_IN-1, max_out=MAX_OUT-1,
        )

    def forward(self, v, e, x, p):
        """
        Examples
        --------
        >>> momentum_update = MomentumUpdate()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(2, 2, 4)
        >>> x = torch.zeros(2, 3, 6)
        >>> p = torch.zeros(2, 3, 7)
        >>> v1, e1, x1, p1 = momentum_update(v, e, x, p)
        >>> list(p1.shape)
        [2, 3, 10]
        >>> assert torch.isclose(v1, v).all()
        >>> assert torch.isclose(e1, e).all()
        """
        p = self.linear(p)
        return v, e, x, p
