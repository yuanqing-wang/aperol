import torch
from ..module import Module
from ..state import State

class VelocityToPositionProjection(Module):
    """ Project velocity to position using a learnable linear transformation.
    
    Examples
    --------
    >>> import torch
    >>> state = State(
    ...     node=torch.randn(5, 16),
    ...     edge=torch.randn(5, 5, 16),
    ...     position=torch.randn(5, 3, 8),
    ...     velocity=torch.randn(5, 3, 7),
    ... )
    >>> proj = VelocityToPositionProjection()
    >>> new_state = proj(state)
    >>> assert new_state.position.shape == (5, 3, 8)
    >>> assert new_state.velocity.shape == (5, 3, 7)
    """
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()
    
    def initialize_parameters(self, state):
        self.weight.materialize((state.velocity.shape[-1], state.position.shape[-1]))
    
    def forward(self, state: State) -> State:
        state.position = state.position + state.velocity @ self.weight
        return state