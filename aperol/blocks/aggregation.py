"""General aggregation methods. """
from functools import partial
import torch
from ..module import Module, Linear

class Aggregation(Module):
    def __init__(self, aggregator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregator = aggregator
        self.linear_x = Linear()
        self.linear_y = Linear()

    def sample(self):
        return self.linear_x.sample()._replace(cls=self.__class__)

    def forward(self, x, y, config=None):
        """Aggregates y onto x and transform.

        Examples
        --------
        >>> x = torch.zeros(5, 3)
        >>> y = torch.ones(5, 5, 2)
        >>> aggregation = Aggregation(torch.sum)
        >>> z = aggregation(x, y)
        >>> z.shape[0]
        5
        """
        if config is None:
            config = self.sample()

        y = self.linear_y(y, config=config)
        y = self.aggregator(y, dim=-2)
        x = self.linear_x(x, config=config)
        return x + y


MeanAggregation = partial(Aggregation, torch.mean)
SumAggregation = partial(Aggregation, torch.sum)

class DotAttentionAggregation(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_k = Linear()
        self.linear_q = Linear()
        self.linear_v = Linear()
        self.linear = Linear()

    def sample(self):
        return self.linear.sample()._replace(cls=self.__class__)

    def forward(self, x, y, config=None):
        """Aggregates y onto x with dot product attention.

        Examples
        --------
        >>> x = torch.zeros(5, 3)
        >>> y = torch.ones(4, 2)
        >>> aggregation = DotAttentionAggregation()
        >>> z = aggregation(x, y)
        >>> z.shape[0]
        5
        """
        if config is None:
            config = self.sample()
        k = self.linear_k(x, config=config)
        q = self.linear_q(y, config=config)
        y = (k @ q.swapaxes(-1, -2)).softmax(-1) @ y # x, y
        y = self.linear_v(y, config=config).sum(-2, keepdims=True)
        x = self.linear(x, config=config)
        return x + y
