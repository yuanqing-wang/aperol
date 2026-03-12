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
        self.weight = torch.nn.UninitializedParameter()
    
    def initialize_parameters(self, state):
        self.weight.materialize((state.edge.shape[-1], state.node.shape[-1]))
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, state: State) -> State:
        # state.edge: (N, N, De)
        # state.node: (N, Dn)
        edge = self.endomorphism(state.edge)  # (N, N, Dn)
        aggregated = self.aggregator(edge, dim=-2)  # (N, Dn)
        aggregated = aggregated @ self.weight  # (N, Dn)
        new_node = state.node + aggregated  # (N, Dn)
        return state.replace(node=new_node)
    
class EdgeToNodeMean(EdgeToNodeAggregation):
    """ Use mean aggregation for edge-to-node aggregation.
    
    Examples
    --------
    >>> import torch
    >>> from ..test_utils import get_random_state, get_simple_endomorphism
    >>> state = get_random_state()
    >>> endomorphism = get_simple_endomorphism()
    >>> mean_agg = EdgeToNodeMean(endomorphism)
    >>> new_state = mean_agg(state)
    >>> assert new_state.node.shape == state.node.shape
    """
    __init__ = partialmethod(EdgeToNodeAggregation.__init__, aggregator=torch.mean)
    
class EdgeToNodeMax(EdgeToNodeAggregation):
    """ Use max aggregation for edge-to-node aggregation.
    
    Examples
    --------
    >>> import torch
    >>> from ..test_utils import get_random_state, get_simple_endomorphism
    >>> state = get_random_state()
    >>> endomorphism = get_simple_endomorphism()
    >>> max_agg = EdgeToNodeMax(endomorphism)
    >>> new_state = max_agg(state)
    >>> assert new_state.node.shape == state.node.shape
    """
    __init__ = partialmethod(
        EdgeToNodeAggregation.__init__, 
        aggregator=lambda x, dim: torch.max(x, dim=dim).values,
    )
    
class AttentionAggregation(Module):
    def __init__(self):
        super().__init__()
        self.k = torch.nn.UninitializedParameter()
        self.q = torch.nn.UninitializedParameter()
        self.v = torch.nn.UninitializedParameter()

    def initialize_parameters(self, x: torch.Tensor, dim: int):
        self.k.materialize((x.shape[-1], x.shape[-1]))
        self.q.materialize((x.shape[-1], x.shape[-1]))
        self.v.materialize((x.shape[-1], x.shape[-1]))
        torch.nn.init.xavier_uniform_(self.k)
        torch.nn.init.xavier_uniform_(self.q)
        torch.nn.init.xavier_uniform_(self.v)
        
    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        k = x @ self.k  # (N, N, De)
        q = x @ self.q  # (N, N, De)
        v = x @ self.v  # (N, N, De)
        att = torch.einsum("...md,...nd->...mn", k, q) # (N, N, N)
        att = torch.softmax(att, dim=dim)  # (N, N, N)
        out = torch.einsum("...mn,...nd->...md", att, v)  # (N, N, De)
        out = out.mean(dim=dim)  # (N, De)
        return out
    
class EdgeToNodeAttention(EdgeToNodeAggregation):
    """ Use attention aggregation for edge-to-node aggregation.
    
    Examples
    --------
    >>> import torch
    >>> from ..test_utils import get_random_state, get_simple_endomorphism
    >>> state = get_random_state()
    >>> endomorphism = get_simple_endomorphism()
    >>> att_agg = EdgeToNodeAttention(endomorphism)
    >>> new_state = att_agg(state)
    >>> assert new_state.node.shape == state.node.shape
    """
    __init__ = partialmethod(EdgeToNodeAggregation.__init__, aggregator=AttentionAggregation())
        
        
