"""Edge to node modules. """
from typing import Optional
from functools import partial
from ..module import Module
from .aggregation import (
    MeanAggregation, SumAggregation, DotAttentionAggregation,
)

class EdgeToNodeAggregation(Module):
    """Aggregate from edge to node. """
    def __init__(self, aggregator: Optional[type] = MeanAggregation):
        super().__init__()
        self.aggregator = aggregator()

    def sample(self):
        return self.aggregator.sample()

    def forward(self, v, e, x, config=None):
        """

        Examples
        --------
        >>> import torch
        >>> edge_to_node_aggregation = EdgeToNodeAggregation()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(3, 4)
        >>> x = torch.zeros(2, 3, 6)
        >>> v, e, x = edge_to_node_aggregation(v, e, x)
        >>> v.shape[0], e.shape[0], x.shape[0]
        (2, 3, 2)
        """
        v = self.aggregator(v, e)
        return v, e, x

MeanEdgeToNodeAggregation = partial(EdgeToNodeAggregation, MeanAggregation)
SumEdgeToNodeAggregation = partial(EdgeToNodeAggregation, SumAggregation)
DotAttentionEdgeToNodeAggregation = partial(
    EdgeToNodeAggregation, DotAttentionAggregation
)
