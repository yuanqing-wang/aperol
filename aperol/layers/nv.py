import torch
from ..module import Module
from ..endomorphism import Endomorphism
from ..state import State

class NodeToVelocityDamping(Module):
    """ Compute the damping factor from node features and apply it to velocity.
    
    Examples
    --------
    >>> import torch
    >>> from ..test_utils import get_random_state, get_simple_endomorphism
    >>> state = get_random_state()
    >>> endomorphism = get_simple_endomorphism()
    >>> damping = NodeToVelocityDamping(endomorphism)
    >>> new_state = damping(state)
    >>> assert new_state.velocity.shape == state.velocity.shape
    
    """
    def __init__(self, endomorphism: Endomorphism):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()
        self.endomorphism = endomorphism
        
    def initialize_parameters(self, state):
        self.weight.materialize((state.node.shape[-1], state.velocity.shape[-1]))
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, state: State) -> State:
        damping_factor = 2 * torch.tanh(self.endomorphism(state.node @ self.weight))  # (N, Dv)
        new_velocity = state.velocity * damping_factor.unsqueeze(-2)
        return state.replace(velocity=new_velocity)
