from functools import partialmethod
import math
from typing import Callable 
import torch
from ..module import Module
from ..state import State
CUTOFF_LOWER = 0.0
CUTOFF_UPPER = 10.0
NUM_BASIS = 50

def get_delta_x(x):
    """Compute the vector difference among geometry.

    Parameters
    ----------
    x : torch.Tensor (N, 3)
        Geometry.

    Returns
    -------
    torch.Tensor(N, N, 3)
        Delta geometry.

    Examples
    --------
    >>> x = torch.randn(5, 3, 8)
    >>> delta_x = get_delta_x(x)
    >>> delta_x.shape
    torch.Size([5, 5, 3, 8])
    """
    return x.unsqueeze(-3) - x.unsqueeze(-4)

def get_distance(x):
    """Compute the distance among geometry.

    Parameters
    ----------
    x : torch.Tensor (N, 3, Dx)
        Geometry.

    Returns
    -------
    torch.Tensor(N, N, 1, Dx)
        Distance.

    Examples
    --------
    >>> x = torch.randn(5, 3, 8)
    >>> delta_x = get_distance(x)
    >>> delta_x.shape
    torch.Size([5, 5, 1, 8])
    """
    delta_x = get_delta_x(x)
    norm = torch.linalg.norm(delta_x, dim=-2, keepdims=True)
    return norm

def cosine_cutoff(x, lower=CUTOFF_LOWER, upper=CUTOFF_UPPER):
    """Cosine cutoff.

    Parameters
    ----------
    x : torch.Tensor (N, N, 1, D)
        Distance.

    Returns
    -------
    torch.Tensor (N, N, 1, D)
        Cutoff indicator.

    Examples
    --------
    >>> x = torch.randn(5, 5, 3, 10)
    >>> x = cosine_cutoff(x)
    >>> x.shape
    torch.Size([5, 5, 3, 10])
    """
    cutoffs = 0.5 * (
        torch.cos(
            math.pi
            * (
                2
                * (x - lower)
                / (upper - lower)
                + 1.0
            )
        )
        + 1.0
    )

    # remove contributions below the cutoff radius
    cutoffs = cutoffs * (x < upper)
    cutoffs = cutoffs * (x > lower)
    return cutoffs

def rbf(x, num_basis=NUM_BASIS, lower=CUTOFF_LOWER, upper=CUTOFF_UPPER):
    """Radial basis function.

    Parameters
    ----------
    x : torch.Tensor (N, N, 1, D)
        Distance

    Returns
    -------
    torch.Tensor (N, N, N_BASIS)

    Examples
    --------
    >>> x = torch.randn(8, 3, 10)
    >>> x_distance = get_distance(x)
    >>> rbf(x_distance).shape
    torch.Size([8, 8, 50, 10])
    """
    offset = torch.linspace(lower, upper, num_basis, device=x.device)
    coeff = -0.5 / (offset[1] - offset[0]) ** 2
    x = x - offset[..., None]
    return torch.exp(coeff * torch.pow(x, 2))

def erbf(x, num_basis=NUM_BASIS, lower=CUTOFF_LOWER, upper=CUTOFF_UPPER):
    """Exponential radial basis funciton.

    Parameters
    ----------
    x : torch.Tensor (N, N, 1, D)
        Distance

    Returns
    -------
    torch.Tensor (N, N, N_BASIS, D)

    Examples
    --------
    >>> x = torch.randn(8, 3, 10)
    >>> x_distance = get_distance(x)
    >>> erbf(x_distance).shape
    torch.Size([8, 8, 50, 10])
    """
    start_value = math.exp(-upper + lower)

    means = torch.linspace(start_value, 1, num_basis, device=x.device)
    alpha = 5.0 / (upper - lower)
    betas = torch.tensor(
        [(2 / num_basis * (1 - start_value)) ** -2] * num_basis,
        device=x.device,
    )
    
    betas, means = betas[..., None], means[..., None]

    return torch.exp(
        -betas * (torch.exp(alpha * (-x + lower)) - means) ** 2
    )
    
class Smearing(Module):
    """Smearing layer for geometry features. """
    def __init__(
        self,
        kernel: Callable,
        num_basis: int = NUM_BASIS,
        lower: float = CUTOFF_LOWER,
        upper: float = CUTOFF_UPPER,
    ):
        super().__init__()
        self.kernel = kernel
        self.num_basis = num_basis
        self.lower = lower
        self.upper = upper
        self.weight_in = torch.nn.UninitializedParameter()
        self.weight_out = torch.nn.UninitializedParameter()
        
    def initialize_parameters(self, state):
        self.weight_in.materialize((state.edge.shape[-1], self.num_basis))
        self.weight_out.materialize((self.num_basis * state.position.shape[-1], state.edge.shape[-1]))
        
    def forward(self, state: State) -> State:
        filter = state.edge @ self.weight_in  # (N, N, num_basis)
        distance = get_distance(state.position)  # (N, N, 1, Dx)
        cutoff_indicator = cosine_cutoff(distance, self.lower, self.upper)  # (N, N, 1, Dx)
        x_smeared = self.kernel(distance, self.num_basis, self.lower, self.upper)  # (N, N, num_basis, Dx)
        x_smeared = x_smeared * cutoff_indicator  # (N, N, num_basis, Dx)
        x_filtered = x_smeared * filter.unsqueeze(-1)  # (N, N, num_basis, Dx)
        x_out = x_filtered.flatten(-2) @ self.weight_out  # (N, N, De)
        new_edge = state.edge + x_out
        return state.replace(edge=new_edge)
    
    
class RBFSmearing(Smearing):
    """ Smearing layer using radial basis function (RBF) kernel.
    
    Examples
    --------
    >>> import torch
    >>> from ..test_utils import get_random_state
    >>> state = get_random_state()
    >>> rbf_smearing = RBFSmearing()
    >>> new_state = rbf_smearing(state)
    >>> assert new_state.edge.shape == state.edge.shape
    
    """
    __init__ = partialmethod(Smearing.__init__, rbf)

class ERBFSmearing(Smearing):
    __init__ = partialmethod(Smearing.__init__, erbf)
