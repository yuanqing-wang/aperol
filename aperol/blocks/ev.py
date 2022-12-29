"""Edge to node modules. """
from typing import Optional
from functools import partialmethod
from ..module import Module, Block
from .aggregation import (
    MeanAggregation, SumAggregation, DotAttentionAggregation,
)
from .xe import get_distance, cosine_cutoff

__all__ = [
    "MeanEdgeToNodeAggregation",
    "SumEdgeToNodeAggregation",
    "DotAttentionEdgeToNodeAggregation",
]

class EdgeToNodeAggregation(Block):
    """Aggregate from edge to node. """
    def __init__(self, aggregator: Optional[type] = MeanAggregation):
        super().__init__()
        self.aggregator = aggregator()

    def sample(self):
        return self.aggregator.sample()._replace(cls=self.__class__)

    def forward(self, v, e, x, config=None):
        """

        Examples
        --------
        >>> import torch
        >>> edge_to_node_aggregation = EdgeToNodeAggregation()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(2, 2, 4)
        >>> x = torch.zeros(2, 3, 6)
        >>> v, e, x = edge_to_node_aggregation(v, e, x)
        >>> v.shape[0], e.shape[0], x.shape[0]
        (2, 2, 2)
        """
        cutoff = cosine_cutoff(get_distance(x)).mean(-1, keepdims=True)
        v = self.aggregator(v, cutoff * e)
        return v, e, x

class MeanEdgeToNodeAggregation(EdgeToNodeAggregation):
    __init__ = partialmethod(EdgeToNodeAggregation.__init__, MeanAggregation)

class SumEdgeToNodeAggregation(EdgeToNodeAggregation):
    __init__ = partialmethod(EdgeToNodeAggregation.__init__, SumAggregation)

class DotAttentionEdgeToNodeAggregation(EdgeToNodeAggregation):
    __init__ = partialmethod(EdgeToNodeAggregation.__init__, DotAttentionAggregation)
