"""Node to edge modules. """
from typing import Optional
from functools import partialmethod
from ..module import Block
from .aggregation import (
    MeanAggregation, SumAggregation, DotAttentionAggregation,
)

__all__ = [
    "MeanNodeToEdgeAggregation",
    "SumNodeToEdgeAggregation",
    # "DotAttentionNodeToEdgeAggregation",
]

class NodeToEdgeAggregation(Block):
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
        >>> edge_to_node_aggregation = NodeToEdgeAggregation()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(2, 2, 4)
        >>> x = torch.zeros(2, 3, 6)
        >>> p = torch.zeros(2, 3, 7)
        >>> v, e, x, p = edge_to_node_aggregation(v, e, x, p)
        >>> v.shape[0], e.shape[0], x.shape[0], p.shape[0]
        (2, 2, 2, 2)
        """
        e = self.aggregator(e, v.unsqueeze(-2).unsqueeze(-2), config=config)
        return v, e, x, p

class MeanNodeToEdgeAggregation(NodeToEdgeAggregation):
    __init__ = partialmethod(NodeToEdgeAggregation.__init__, MeanAggregation)

class SumNodeToEdgeAggregation(NodeToEdgeAggregation):
    __init__ = partialmethod(NodeToEdgeAggregation.__init__, SumAggregation)

class DotAttentionNodeToEdgeAggregation(NodeToEdgeAggregation):
    __init__ = partialmethod(
        NodeToEdgeAggregation.__init__, DotAttentionAggregation
    )
