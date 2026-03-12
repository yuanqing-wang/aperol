from functools import partialmethod
import math
from turtle import forward
from typing import Callable 
import torch
from ..module import Module
from ..state import State

class PositionToVelocityKick(Module):
    """ Compute the position-to-velocity kick.
    
    Examples
    --------
    >>> from ..test_utils import get_random_state
    >>> state = get_random_state()
    >>> kick = PositionToVelocityKick()
    >>> new_state = kick(state)
    >>> assert new_state.velocity.shape == state.velocity.shape
    """
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()
        
    def initialize_parameters(self, state):
        self.weight.materialize((state.position.shape[-1], state.velocity.shape[-1]))
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, state: State) -> State:
        delta_x = state.position.unsqueeze(-3) - state.position.unsqueeze(-4)  # (N, N, 3, Dx)
        new_velocity = state.velocity + (delta_x @ self.weight).mean(dim=-3)  # (N, Dv)
        return state.replace(velocity=new_velocity)
