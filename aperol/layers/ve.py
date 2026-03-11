import torch
from ..module import Module
from ..state import State

class VelocityDotToEdge(Module):
    """ Compute the dot product of velocity and add it to edge features.
    
    Examples
    --------
    >>> import torch
    >>> state = State(
    ...     node=torch.randn(5, 16),
    ...     edge=torch.randn(5, 5, 16),
    ...     position=torch.randn(5, 3, 8),
    ...     velocity=torch.randn(5, 3, 7),
    ... )
    >>> dot_to_edge = VelocityDotToEdge()
    >>> new_state = dot_to_edge(state)
    >>> assert new_state.edge.shape == (5, 5, 16)
    
    """
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()
        
    def initialize_parameters(self, state):
        self.weight.materialize((state.velocity.shape[-1], state.edge.shape[-1]))
        
    def forward(self, state: State) -> State:
        vv = torch.einsum("ntd,mtd->mnd", state.velocity, state.velocity)  # (N, N, Dv)
        state.edge = state.edge + vv @ self.weight  # (N, N, De)
        return state