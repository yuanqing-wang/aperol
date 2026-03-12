import torch
from ..module import Module
from ..endomorphism import Endomorphism
from ..state import State

class VelocityDotToEdge(Module):
    """ Compute the dot product of velocity and add it to edge features.
    
    Examples
    --------
    >>> import torch
    >>> from ..test_utils import get_random_state, get_simple_endomorphism
    >>> state = get_random_state()
    >>> endomorphism = get_simple_endomorphism()
    >>> dot_to_edge = VelocityDotToEdge(endomorphism)
    >>> new_state = dot_to_edge(state)
    >>> assert new_state.edge.shape == state.edge.shape
    
    """
    def __init__(self, endomorphism: Endomorphism):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()
        self.endomorphism = endomorphism
        
    def initialize_parameters(self, state):
        self.weight.materialize((state.velocity.shape[-1], state.edge.shape[-1]))
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, state: State) -> State:
        vv = torch.einsum("ntd,mtd->mnd", state.velocity, state.velocity)  # (N, N, Dv)
        new_edge = state.edge + self.endomorphism(vv @ self.weight)  # (N, N, De)
        return state.replace(edge=new_edge)
