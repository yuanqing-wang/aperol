import torch
from ..module import Module
from ..state import State

class VelocityNormToNode(Module):
    """ Compute the norm of velocity and add it to node features.
    
    Examples
    --------
    >>> from ..test_utils import get_random_state
    >>> state = get_random_state()
    >>> norm_to_node = VelocityNormToNode()
    >>> new_state = norm_to_node(state)
    >>> assert new_state.node.shape == state.node.shape
        
    """
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()
        
    def initialize_parameters(self, state):
        self.weight.materialize((state.velocity.shape[-1], state.node.shape[-1]))
    
    def forward(self, state: State) -> State:
        norm = torch.norm(state.velocity, dim=-2)  # (N, Dv)
        new_node = state.node + norm @ self.weight  # (N, D)
        return state.replace(node=new_node)
        
