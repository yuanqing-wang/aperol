"""Momentum to edge update. """
from functools import partialmethod
from typing import Callable, NamedTuple, Optional
import torch
import math
from ..module import Block, Linear
from ..constants import NUM_BASIS, CUTOFF_LOWER, CUTOFF_UPPER

__all__ = ["SpatialAttention"]

class SpatialAttention(Block):
    """Spatial attention module. """
    def __init__(self):
        super().__init__()
        self.linear_k = Linear(activation=None, bias=False)
        self.linear_q = Linear(activation=None, bias=False)
        self.linear_summarize = Linear()
        self.linear = Linear()

    def sample(self):
        return self.linear.sample()._replace(cls=self.__class__)

    def forward(
            self, 
            v: torch.Tensor, e: torch.Tensor, x: torch.Tensor, p: torch.Tensor,
            config: Optional[NamedTuple] = None,
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
        if config is None:
            config = self.sample()

        x_k = self.linear_k(p, config=config)
        x_q = self.linear_q(p, config=config)
        a = torch.linalg.norm(x_k.unsqueeze(-3) - x_q.unsqueeze(-4), dim=-2)
        a = self.linear_summarize(a, config=config)
        e = self.linear(e, config=config)
        e = a + e
        return v, e, x, p