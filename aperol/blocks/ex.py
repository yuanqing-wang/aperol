"""Edge to geometry modules. """

from ..module import Module, Linear, BlockModule
from .xe import get_delta_x
import torch

class EdgeToGeometryUpdate(BlockModule):
    """Update from edge to geometry. """
    def __init__(self):
        super().__init__()
        self.linear = Linear(activation=None, bias=False)

    def sample(self):
        return self.linear.sample()._replace(cls=self.__class__)

    def forward(self, v, e, x, config=None):
        """
        Examples
        --------
        >>> edge_to_geometry_update = EdgeToGeometryUpdate()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(2, 2, 8)
        >>> x = torch.zeros(2, 3, 6)
        >>> config = edge_to_geometry_update.Config(10)
        >>> v1, e1, x1 = edge_to_geometry_update(v, e, x, config=config)
        """
        config = self.linear.Config(x.shape[-1])
        delta_x = get_delta_x(x)
        delta_x = self.linear(e, config=config).unsqueeze(-2) * delta_x
        delta_x = delta_x.mean(0)
        x = delta_x + x
        return v, e, x
