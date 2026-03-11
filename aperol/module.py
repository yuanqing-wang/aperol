import abc
import re
import torch
from torch.nn.modules.lazy import LazyModuleMixin
from .state import State

class Module(LazyModuleMixin, torch.nn.Module):
    """Base module for `aperol` building blocks."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def initialize_parameters(self, state):
        """Initialize parameters based on the input state."""
        return None

    @abc.abstractmethod
    def forward(
        self,
        state: State
    ) -> State:
        """Forward pass.

        Parameters
        ----------
        state : State
            The current state of the system.

        Returns
        -------
        State
            The updated state of the system.
        """
        raise NotImplementedError


