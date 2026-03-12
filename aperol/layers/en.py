from typing import Callable
from functools import partialmethod

import torch
from ..module import Module
from ..endomorphism import Endomorphism
from ..state import State

class EdgeToNodeAggregation(Module):
    def __init__(self, endomorphism: Endomorphism, aggregator: Callable):
        super().__init__()
        self.endomorphism = endomorphism
        self.aggregator = aggregator
        
    def forward(self, state: State) -> State:
        # state.edge: (N, N, De)
        # state.node: (N, Dn)
        edge = self.endomorphism(state.edge)  # (N, N, Dn)
        aggregated = self.aggregator(edge, dim=2)  # (N, Dn)
        new_node = state.node + aggregated  # (N, Dn)
        return state.replace(node=new_node)
    
class EdgeToNodeMean(EdgeToNodeAggregation):
    __init__ = partialmethod(EdgeToNodeAggregation.__init__, aggregator=torch.mean)
    
class EdgeToNodeMax(EdgeToNodeAggregation):
    __init__ = partialmethod(EdgeToNodeAggregation.__init__, aggregator=torch.max)