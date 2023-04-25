"""Geometry to node modules. """

import torch
from ..module import Block, Linear

__all__ = ["DotProductReduce"]


class DotProductReduce(Block):
    def __init__(self):
        super().__init__()
        self.linear_k = Linear(activation=None, bias=False)
        self.linear_q = Linear(activation=None, bias=False)
        self.linear_summarize = Linear()
        self.linear = Linear()

    def sample(self):
        return self.linear.sample()._replace(cls=self.__class__)

    def forward(self, v, e, x, p, config=None):
        """
        Examples
        --------
        >>> reduce = DotProductReduce()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(2, 2, 8)
        >>> x = torch.zeros(2, 3, 6)
        >>> v, e, x = reduce(v, e, x)
        """
        if config is None:
            config = self.sample()

        # (N, 3, D)
        k = self.linear_k(p, config=config)
        q = self.linear_q(p, config=config)

        # (N, D)
        kq = (k * q).sum(-2)
        kq = self.linear_summarize(kq, config=config)

        # (N, D)
        v = self.linear(v, config=config) + kq

        return v, e, x, p
