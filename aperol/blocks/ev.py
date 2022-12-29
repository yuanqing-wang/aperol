"""Edge to node modules. """
from typing import Optional
from functools import partial
from ..module import Module, BlockModule
from .aggregation import (
    MeanAggregation, SumAggregation, DotAttentionAggregation,
)
from .xe import get_distance, cosine_cutoff

class EdgeToNodeAggregation(BlockModule):
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

MeanEdgeToNodeAggregation = partial(EdgeToNodeAggregation, MeanAggregation)
SumEdgeToNodeAggregation = partial(EdgeToNodeAggregation, SumAggregation)
DotAttentionEdgeToNodeAggregation = partial(
    EdgeToNodeAggregation, DotAttentionAggregation
)
