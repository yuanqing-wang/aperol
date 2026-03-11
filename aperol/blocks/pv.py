"""Geometry to node modules. """

import torch
from ..module import Module, Linear

__all__ = ["DotProductReduce"]


class DotProductReduce(Module):
    def __init__(self):
        super().__init__()
        self.linear_k = Linear(activation=None, bias=False)
        self.linear_q = Linear(activation=None, bias=False)
        self.linear_summarize = Linear()
        self.linear = Linear()

    def forward(self, v, e, x, p):
        """
        Examples
        --------
        >>> reduce = DotProductReduce()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(2, 2, 8)
        >>> x = torch.zeros(2, 3, 6)
        >>> p = torch.zeros(2, 3, 7)
        >>> v, e, x, p = reduce(v, e, x, p)
        """
        # (N, 3, D)
        k = self.linear_k(p)
        q = self.linear_q(p)

        # (N, D)
        kq = (k * q).sum(-2)
        kq = self.linear_summarize(kq)

        # (N, D)
        v = self.linear(v) + kq

        return v, e, x, p
