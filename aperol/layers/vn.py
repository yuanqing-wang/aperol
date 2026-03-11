import torch
from ..module import Module
from ..state import State

class VelocityNormToNode(Module):
    """ Compute the norm of velocity and add it to node features.
    
    Examples
    --------
    >>> import torch
    >>> state = State(
    ...     node=torch.randn(5, 16),
    ...     edge=torch.randn(5, 5, 16),
    ...     position=torch.randn(5, 3, 8),
    ...     velocity=torch.randn(5, 3, 7),
    ... )
    >>> norm_to_node = VelocityNormToNode()
    >>> new_state = norm_to_node(state)
    >>> assert new_state.node.shape == (5, 16)
        
    """
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()
        
    def initialize_parameters(self, state):
        self.weight.materialize((state.velocity.shape[-1], state.node.shape[-1]))
    
    def forward(self, state: State) -> State:
        norm = torch.norm(state.velocity, dim=-2)  # (N, Dv)
        state.node = state.node + norm @ self.weight  # (N, D)
        return state
        
