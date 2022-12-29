"""Geometry to node modules. """

import torch
from ..module import Module, Linear

class DotProductReduce(Module):
    def __init__(self):
        super().__init__()
        self.linear_k = Linear(activation=None, bias=False)
        self.linear_q = Linear(activation=None, bias=False)
        self.linear_summarize = Linear()
        self.linear = Linear()

    def sample(self):
        return self.linear.sample()._replace(cls=self.__class__)

    def forward(self, v, e, x, config=None):
        if config is None:
            config = self.sample()

        # (N, 3, D)
        x_eq = x[..., 1:]

        # (N, 3, D)
        k = self.linear_k(x_eq)
        q = self.linear_q(x_eq)

        # (N, D)
        kq = torch.einsum("abc, adc -> ac", k, q)
        kq = self.linear_summarize(kq)

        # (N, D)
        v = self.linear(v) + kq

        return v, e, x
