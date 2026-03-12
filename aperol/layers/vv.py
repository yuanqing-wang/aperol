from typing import Optional

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
        features: Optional[int] = None,
    ):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()
        self.features = features
        
    def initialize_parameters(self, state):
        in_features = state.velocity.shape[-1]
        out_features = self.features if self.features is not None else in_features
        self.weight.materialize((in_features, out_features))
    
    def forward(self, state: State) -> State:
        new_velocity = state.velocity @ self.weight
        return state.replace(velocity=new_velocity)
