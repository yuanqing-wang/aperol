from aperol.endomorphism import Endomorphism
import torch
from ..module import Module
from ..state import State

class VelocityNormToNode(Module):
    """ Compute the norm of velocity and add it to node features.
    
    Examples
    --------
    >>> from ..test_utils import get_random_state, get_simple_endomorphism
    >>> state = get_random_state()
    >>> endomorphism = get_simple_endomorphism()
    >>> norm_to_node = VelocityNormToNode(endomorphism)
    >>> new_state = norm_to_node(state)
    >>> assert new_state.node.shape == state.node.shape
        
    """
    def __init__(self, endomorphism: Endomorphism):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()
        self.endomorphism = endomorphism
        
    def initialize_parameters(self, state):
        self.weight.materialize((state.velocity.shape[-1], state.node.shape[-1]))
        torch.nn.init.xavier_uniform_(self.weight)
    
    def forward(self, state: State) -> State:
        norm = torch.norm(state.velocity, dim=-2)  # (N, Dv)
        new_node = state.node + self.endomorphism(norm @ self.weight)  # (N, Dn)
        return state.replace(node=new_node)
        
