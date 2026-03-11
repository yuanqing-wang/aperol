import torch
from ..module import Module
from ..state import State

class VelocityProjection(Module):
    """ Project velocity to a new space using a learnable linear transformation.
    
    Parameters
    ----------
    features : int
        The number of output features for the velocity projection.
        
    Examples
    --------
    >>> from ..test_utils import get_random_state
    >>> state = get_random_state()
    >>> proj = VelocityProjection(features=10)
    >>> new_state = proj(state)
    >>> assert new_state.velocity.shape == (state.velocity.shape[0], state.velocity.shape[1], 10)
        
    """
    def __init__(
        self,
        features: int,
    ):
        super().__init__()
        self.linear = torch.nn.LazyLinear(features, bias=False)
    
    def forward(self, state: State) -> State:
        new_velocity = self.linear(state.velocity)
        return state.replace(velocity=new_velocity)
