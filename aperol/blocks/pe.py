"""Momentum to edge update. """
import torch
from ..module import Module, Linear

__all__ = ["SpatialAttention"]

class SpatialAttention(Module):
    """Spatial attention module. """
    def __init__(self):
        super().__init__()
        self.linear_k = Linear(activation=None, bias=False)
        self.linear_q = Linear(activation=None, bias=False)
        self.linear_summarize = Linear()
        self.linear = Linear()

    def forward(
            self, 
            v: torch.Tensor, e: torch.Tensor, x: torch.Tensor, p: torch.Tensor,
        ):
        """

        Examples
        --------
        >>> spatial_attention = SpatialAttention()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(2, 2, 4)
        >>> x = torch.zeros(2, 3, 6)
        >>> p = torch.zeros(2, 3, 7)
        >>> v, e, x, p = spatial_attention(v, e, x, p)
        """
        x_k = self.linear_k(p)
        x_q = self.linear_q(p)
        a = torch.linalg.norm(x_k.unsqueeze(-3) - x_q.unsqueeze(-4), dim=-2)
        a = self.linear_summarize(a)
        e = self.linear(e)
        e = a + e
        return v, e, x, p
