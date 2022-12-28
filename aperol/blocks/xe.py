"""Geometry to edge modules. """
from typing import Callable, NamedTuple, Optional
import torch
import math
from ..module import Module, Linear
from ..constants import NUM_BASIS, CUTOFF_LOWER, CUTOFF_UPPER

def get_delta_x(x):
    """Compute the vector difference among geometry.

    Parameters
    ----------
    x : torch.Tensor (N, 3, D)
        Geometry.

    Returns
    -------
    torch.Tensor(N, N, 3, D)
        Delta geometry.

    Examples
    --------
    >>> x = torch.randn(5, 3, 10)
    >>> delta_x = get_delta_x(x)
    >>> delta_x.shape
    torch.Size([5, 5, 3, 10])
    """
    return x.unsqueeze(0) - x.unsqueeze(1)

def get_distance(x):
    """Compute the distance among geometry.

    Parameters
    ----------
    x : torch.Tensor (N, 3, D)
        Geometry.

    Returns
    -------
    torch.Tensor(N, N, 1, D)
        Distance.

    Examples
    --------
    >>> x = torch.randn(5, 3, 10)
    >>> delta_x = get_distance(x)
    >>> delta_x.shape
    torch.Size([5, 5, 1, 10])
    """
    delta_x = get_delta_x(x)
    return torch.linalg.norm(delta_x, dim=2, keepdims=True)

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
    torch.Tensor (N, N, N_BASIS, D)

    Examples
    --------
    >>> x = torch.randn(8, 3, 10)
    >>> x_distance = get_distance(x)
    >>> rbf(x_distance).shape
    torch.Size([8, 8, 50, 10])
    """
    offset = torch.linspace(lower, upper, num_basis)
    coeff = -0.5 / (offset[1] - offset[0]) ** 2
    x = x - offset.unsqueeze(-1)
    return torch.exp(coeff * torch.pow(x, 2))

def erbf(x, num_basis=NUM_BASIS, lower=0.0, upper=5.0):
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
    start_value = torch.exp(
        torch.scalar_tensor(-upper + lower)
    )

    means = torch.linspace(start_value, 1, num_basis).unsqueeze(-1)
    alpha = 5.0 / (upper - lower)
    betas = torch.tensor(
        [(2 / num_basis * (1 - start_value)) ** -2] * num_basis
    ).unsqueeze(-1)

    return torch.exp(
        -betas * (torch.exp(alpha * (-x + lower)) - means) ** 2
    )

class Smearing(Module):
    """Smear the distance into multi-dimensional vectors. """
    def __init__(self, kernel: Callable = rbf):
        super().__init__()
        self.kernel = kernel
        self.filter_generation = Linear(
            max_out=NUM_BASIS,
            activation=None, bias=False,
        )
        self.filter_combine = Linear(
            max_in=NUM_BASIS,
            activation=None, bias=None,
        )
        self.linear = Linear()

    def sample(self):
        return self.linear.sample()

    def forward(
            self, v: torch.Tensor, e: torch.Tensor, x: torch.Tensor,
            config: Optional[NamedTuple] = None,
        ):
        """Smear distances with a kernel.

        Examples
        --------
        >>> smearing = Smearing()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(2, 2, 8)
        >>> x = torch.zeros(2, 3, 6)
        >>> v, e, x = smearing(v, e, x)
        """
        if config is None:
            config = self.sample()

        # fix the out_features for filter generation
        filter_config = self.filter_generation.Config(NUM_BASIS)

        # (N, N, N_BASIS, 1)
        filter = self.filter_generation(e, filter_config).unsqueeze(-1)

        # (N, N, 1, N_CHANNELS)
        delta_x_norm = get_distance(x)

        # (N, N, 1, N_CHANNELS)
        cutoff_indicator = cosine_cutoff(delta_x_norm)

        # (N, N, N_BASIS, N_CHANNELS)
        x_smeared = self.kernel(delta_x_norm)
        x_smeared = x_smeared * cutoff_indicator

        # (N, N, N_BASIS)
        x_filtered = (filter * x_smeared).mean(-1)

        # (N, N, out_features)
        e = self.filter_combine(x_filtered, config=config)\
            + self.linear(e, config=config)

        return v, e, x
