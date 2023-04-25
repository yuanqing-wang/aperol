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
    # "DotAttentionEdgeToNodeAggregation",
]

class EdgeToNodeAggregation(Block):
    """Aggregate from edge to node. """
    def __init__(self, aggregator: Optional[type] = MeanAggregation):
        super().__init__()
        self.aggregator = aggregator()

    def sample(self):
        return self.aggregator.sample()._replace(cls=self.__class__)

    def forward(self, v, e, x, p, config=None):
        """

        Examples
        --------
        >>> import torch
        >>> edge_to_node_aggregation = EdgeToNodeAggregation()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(2, 2, 4)
        >>> x = torch.zeros(2, 3, 6)
        >>> p = torch.zeros(2, 3, 7)
        >>> v, e, x, p = edge_to_node_aggregation(v, e, x, p)
        >>> v.shape[0], e.shape[0], x.shape[0], p.shape[0]
        (2, 2, 2, 2)
        """
        if config is None:
            config = self.sample()
        cutoff = cosine_cutoff(get_distance(x[..., 0]))
        v = self.aggregator(v, cutoff * e, config=config)
        return v, e, x, p

class MeanEdgeToNodeAggregation(EdgeToNodeAggregation):
    __init__ = partialmethod(EdgeToNodeAggregation.__init__, MeanAggregation)

class SumEdgeToNodeAggregation(EdgeToNodeAggregation):
    __init__ = partialmethod(EdgeToNodeAggregation.__init__, SumAggregation)

class DotAttentionEdgeToNodeAggregation(EdgeToNodeAggregation):
    __init__ = partialmethod(EdgeToNodeAggregation.__init__, DotAttentionAggregation)
