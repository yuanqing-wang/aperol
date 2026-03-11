import torch
from ..module import Module
from ..state import State

class VelocityDotToEdge(Module):
    """ Compute the dot product of velocity and add it to edge features.
    
    Examples
    --------
    >>> import torch
    >>> from ..test_utils import get_random_state
    >>> state = get_random_state()
    >>> dot_to_edge = VelocityDotToEdge()
    >>> new_state = dot_to_edge(state)
    >>> assert new_state.edge.shape == state.edge.shape
    
    """
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()
        
    def initialize_parameters(self, state):
        self.weight.materialize((state.velocity.shape[-1], state.edge.shape[-1]))
        
    def forward(self, state: State) -> State:
        vv = torch.einsum("ntd,mtd->mnd", state.velocity, state.velocity)  # (N, N, Dv)
        new_edge = state.edge + vv @ self.weight  # (N, N, De)
        return state.replace(edge=new_edge)
