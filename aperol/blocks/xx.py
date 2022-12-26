"""Geometry modules. """
import torch
from ..module import Module, Linear

class GeometryUpdate(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(activation=None, bias=False)

    def forward(self, v, e, x, config=None):
        """
        Examples
        --------
        >>> geometry_update = GeometryUpdate()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(3, 4)
        >>> x = torch.zeros(2, 3, 6)
        >>> config = geometry_update.Config(10)
        >>> v1, e1, x1 = geometry_update(v, e, x, config=config)
        >>> list(x1.shape)
        [2, 3, 10]
        >>> assert torch.isclose(v1, v).all()
        >>> assert torch.isclose(e1, e).all()
        """
        if config is None:
            config = self.sample()
        x = self.linear(x, config=config)
        return v, e, x
