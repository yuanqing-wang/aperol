from functools import partial
import torch
from ..module import Module, ParametrizedModule, Linear

class Aggregation(ParametrizedModule):
    def __init__(self, aggregator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregator = aggregator
        self.linear_x = Linear()
        self.linear_y = Linear()

    def forward(self, x, y):
        """Aggregates y onto x and transform. """
        y = self.aggregator(y, dim=0, keepdims=True)
        y = self.linear_y(y)
        x = self.linear_x(x)
        return x + y


MeanAggregation = partial(Aggregation, torch.mean)
SumAggregation = partial(Aggregation, torch.sum)

class DotAttentionAggregation(ParametrizedModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_k = Linear()
        self.linear_q = Linear()
        self.linear_v = Linear()

    def forward(self, x, y):
        """Aggregates y onto x with dot product attention. """
        k = self.linear_k(x)
        q = self.linear_q(y)
        a = (k @ q.t()).softmax(-1) # x, y
        x = self.linear_v(a @ x) + x
        return x
