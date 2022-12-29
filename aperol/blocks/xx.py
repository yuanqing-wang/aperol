"""Geometry modules. """
import torch
from ..module import Module, Linear
from ..constants import MAX_IN, MAX_OUT

class GeometryUpdate(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(
            activation=None, bias=False, max_in=MAX_IN-1, max_out=MAX_OUT-1,
        )

    def forward(self, v, e, x, config=None):
        """
        Examples
        --------
        >>> geometry_update = GeometryUpdate()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(2, 2, 4)
        >>> x = torch.zeros(2, 3, 6)
        >>> config = geometry_update.Config(10)
        >>> v1, e1, x1 = geometry_update(v, e, x, config=config)
        >>> list(x1.shape)
        [2, 3, 11]
        >>> assert torch.isclose(v1, v).all()
        >>> assert torch.isclose(e1, e).all()
        """
        if config is None:
            config = self.sample()
        x0 = x[..., :1]
        x1 = x[..., 1:]
        x1 = self.linear(x1, config=config)
        x = torch.cat([x0, x1], dim=-1)
        return v, e, x

class GeometryReduce(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(
            activation=None, bias=False, max_in=MAX_IN-1, max_out=1, min_out=0,
        )

    def forward(self, v, e, x, config=None):
        """
        Examples
        --------
        >>> geometry_reduce = GeometryReduce()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(2, 2, 4)
        >>> x = torch.zeros(2, 3, 6)
        >>> v1, e1, x1 = geometry_reduce(v, e, x)
        >>> list(x1.shape)
        [2, 3, 6]
        >>> assert torch.isclose(v1, v).all()
        >>> assert torch.isclose(e1, e).all()
        """
        config = self.linear.Config(1)
        x0 = x[..., :1]
        x1 = x[..., 1:]
        delta_x0 = self.linear(x1, config=config)
        x0 = x0 + delta_x0
        x = torch.cat([x0, x1], dim=-1)
        return v, e, x
