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
    >>> import torch
    >>> state = State(
    ...     node=torch.randn(5, 16),
    ...     edge=torch.randn(5, 5, 16),
    ...     position=torch.randn(5, 3, 8),
    ...     velocity=torch.randn(5, 3, 7),
    ... )
    >>> proj = VelocityProjection(features=10)
    >>> new_state = proj(state)
    >>> assert new_state.velocity.shape == (5, 3, 10)
        
    """
    def __init__(
        self,
        features: int,
    ):
        super().__init__()
        self.linear = torch.nn.LazyLinear(features, bias=False)
    
    def forward(self, state: State) -> State:
        state.velocity = self.linear(state.velocity)
        return state
