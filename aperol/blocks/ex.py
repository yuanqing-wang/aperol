"""Edge to geometry modules. """

from ..module import Module, Linear, Block
from .xe import get_delta_x
import torch

__all__ = [
    "EdgeToGeometryUpdate",
]

class EdgeToGeometryUpdate(Block):
    """Update from edge to geometry. """
    def __init__(self):
        super().__init__()
        self.linear = Linear(activation=None, bias=False)

    def sample(self):
        return self.linear.sample()._replace(cls=self.__class__)

    def forward(self, v, e, x, p, config=None):
        """
        Examples
        --------
        >>> edge_to_geometry_update = EdgeToGeometryUpdate()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(2, 2, 8)
        >>> x = torch.zeros(2, 3, 6)
        >>> p = torch.zeros(2, 3, 7)
        >>> config = edge_to_geometry_update.Config(10)
        >>> v1, e1, x1, p1 = edge_to_geometry_update(v, e, x, p, config=config)
        """
        config = self.linear.Config(x.shape[-1])
        delta_x = x.unsqueeze(-3) - x.unsqueeze(-4)
        delta_x = self.linear(e, config=config).unsqueeze(-2) * delta_x
        delta_x = delta_x.mean(-3)
        x = delta_x + x
        return v, e, x, p
