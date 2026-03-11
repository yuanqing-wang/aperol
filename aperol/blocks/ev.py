"""Edge to node modules. """
from typing import Optional
from functools import partialmethod
from ..module import Module
from .aggregation import (
    MeanAggregation, SumAggregation, DotAttentionAggregation,
)
from .xe import get_distance, cosine_cutoff

__all__ = [
    "MeanEdgeToNodeAggregation",
    "SumEdgeToNodeAggregation",
    "DotAttentionEdgeToNodeAggregation",
]

class EdgeToNodeAggregation(Module):
    """Aggregate from edge to node. """
    def __init__(self, aggregator: Optional[type] = MeanAggregation):
        super().__init__()
        self.aggregator = aggregator()

    def forward(self, v, e, x, p):
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
        cutoff = cosine_cutoff(get_distance(x[..., 0]))
        v = self.aggregator(v, cutoff * e)
        return v, e, x, p

class MeanEdgeToNodeAggregation(EdgeToNodeAggregation):
    __init__ = partialmethod(EdgeToNodeAggregation.__init__, MeanAggregation)

class SumEdgeToNodeAggregation(EdgeToNodeAggregation):
    __init__ = partialmethod(EdgeToNodeAggregation.__init__, SumAggregation)

class DotAttentionEdgeToNodeAggregation(EdgeToNodeAggregation):
    __init__ = partialmethod(EdgeToNodeAggregation.__init__, DotAttentionAggregation)
