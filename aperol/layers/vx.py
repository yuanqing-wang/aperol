import torch
from ..module import Module
from ..state import State

class VelocityToPositionProjection(Module):
    """ Project velocity to position using a learnable linear transformation.
    
    Examples
    --------
    >>> from ..test_utils import get_random_state
    >>> state = get_random_state()
    >>> proj = VelocityToPositionProjection()
    >>> new_state = proj(state)
    >>> assert new_state.position.shape == state.position.shape
    >>> assert new_state.velocity.shape == state.velocity.shape
    """
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()
    
    def initialize_parameters(self, state):
        self.weight.materialize((state.velocity.shape[-1], state.position.shape[-1]))
    
    def forward(self, state: State) -> State:
        new_position = state.position + state.velocity @ self.weight
        return state.replace(position=new_position)
