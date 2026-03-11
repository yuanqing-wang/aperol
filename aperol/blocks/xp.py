"""Geometry modules. """
import torch
from ..module import Module, Linear
from ..constants import MAX_IN, MAX_OUT

__all__ = ["GeometryReduce"]

class GeometryReduce(Module):
    def __init__(
        self,
        hidden_features: int,
    ):
        super().__init__()
        self.linear = torch.nn.LazyLinear(out_features=hidden_features, bias=False)

    def forward(self, v, e, x, p):
        """
        Examples
        --------
        >>> geometry_reduce = GeometryReduce()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(2, 2, 4)
        >>> x = torch.zeros(2, 3, 6)
        >>> p = torch.zeros(2, 3, 7)
        >>> v1, e1, x1, p1 = geometry_reduce(v, e, x, p)
        >>> list(x1.shape)
        [2, 3, 6]
        >>> assert torch.isclose(v1, v).all()
        >>> assert torch.isclose(e1, e).all()
        """
        delta_x = self.linear(p)
        x = x + delta_x
        return v, e, x, p
