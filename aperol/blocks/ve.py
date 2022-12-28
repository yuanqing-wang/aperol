"""Node to edge modules. """
from typing import Optional
from functools import partial
from ..module import Module
from .aggregation import (
    MeanAggregation, SumAggregation, DotAttentionAggregation,
)

class NodeToEdgeAggregation(Module):
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
        >>> edge_to_node_aggregation = NodeToEdgeAggregation()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(2, 2, 4)
        >>> x = torch.zeros(2, 3, 6)
        >>> v, e, x = edge_to_node_aggregation(v, e, x)
        >>> v.shape[0], e.shape[0], x.shape[0]
        (2, 2, 2)
        """
        e = self.aggregator(e, v.unsqueeze(0))
        return v, e, x

MeanNodeToEdgeAggregation = partial(NodeToEdgeAggregation, MeanAggregation)
SumNodeToEdgeAggregation = partial(NodeToEdgeAggregation, SumAggregation)
DotAttentionNodeToEdgeAggregation = partial(
    NodeToEdgeAggregation, DotAttentionAggregation
)
