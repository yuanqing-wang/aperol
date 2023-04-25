"""Geometry to edge modules. """
from functools import partialmethod
from typing import Callable, NamedTuple, Optional
import torch
import math
from ..module import Block, Linear
from ..constants import NUM_BASIS, CUTOFF_LOWER, CUTOFF_UPPER

__all__ = ["RBFSmearing", "ERBFSmearing", "SpatialAttention"]

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
    >>> x = torch.randn(5, 3)
    >>> delta_x = get_delta_x(x)
    >>> delta_x.shape
    torch.Size([5, 5, 3])
    """
    return x.unsqueeze(-2) - x.unsqueeze(-3)

def get_distance(x):
    """Compute the distance among geometry.

    Parameters
    ----------
    x : torch.Tensor (N, 3)
        Geometry.

    Returns
    -------
    torch.Tensor(N, N, 1)
        Distance.

    Examples
    --------
    >>> x = torch.randn(5, 3)
    >>> delta_x = get_distance(x)
    >>> delta_x.shape
    torch.Size([5, 5, 1])
    """
    delta_x = get_delta_x(x)
    norm = torch.linalg.norm(delta_x, dim=-1, keepdims=True)
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
    >>> x = torch.randn(8, 3)
    >>> x_distance = get_distance(x)
    >>> rbf(x_distance).shape
    torch.Size([8, 8, 50])
    """
    offset = torch.linspace(lower, upper, num_basis)
    coeff = -0.5 / (offset[1] - offset[0]) ** 2
    x = x - offset
    return torch.exp(coeff * torch.pow(x, 2))

def erbf(x, num_basis=NUM_BASIS, lower=0.0, upper=5.0):
    """Exponential radial basis funciton.

    Parameters
    ----------
    x : torch.Tensor (N, N, 1)
        Distance

    Returns
    -------
    torch.Tensor (N, N, N_BASIS)

    Examples
    --------
    >>> x = torch.randn(8, 3)
    >>> x_distance = get_distance(x)
    >>> erbf(x_distance).shape
    torch.Size([8, 8, 50])
    """
    start_value = torch.exp(
        torch.scalar_tensor(-upper + lower)
    )

    means = torch.linspace(start_value, 1, num_basis)
    alpha = 5.0 / (upper - lower)
    betas = torch.tensor(
        [(2 / num_basis * (1 - start_value)) ** -2] * num_basis
    )

    return torch.exp(
        -betas * (torch.exp(alpha * (-x + lower)) - means) ** 2
    )

class Smearing(Block):
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
        return self.linear.sample()._replace(cls=self.__class__)

    def forward(
            self, 
            v: torch.Tensor, e: torch.Tensor, x: torch.Tensor, p: torch.Tensor,
            config: Optional[NamedTuple] = None,
        ):
        """Smear distances with a kernel.

        Examples
        --------
        >>> smearing = Smearing()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(2, 2, 8)
        >>> x = torch.zeros(2, 3, 6)
        >>> p = torch.zeros(2, 3, 7)
        >>> v, e, x, p = smearing(v, e, x, p)
        """
        if config is None:
            config = self.sample()

        # fix the out_features for filter generation
        filter_config = self.filter_generation.Config(NUM_BASIS)

        # (N, N, N_BASIS)
        filter = self.filter_generation(e, filter_config)

        # (N, N, 1)
        delta_x_norm = get_distance(x[..., 0])

        # (N, N, 1)
        cutoff_indicator = cosine_cutoff(delta_x_norm)

        # (N, N, N_BASIS)
        x_smeared = self.kernel(delta_x_norm)
        x_smeared = x_smeared * cutoff_indicator

        # (N, N, N_BASIS)
        x_filtered = (filter * x_smeared)

        # (N, N, out_features)
        e = self.filter_combine(x_filtered, config=config)\
            + self.linear(e, config=config)

        return v, e, x, p

class RBFSmearing(Smearing):
    __init__ = partialmethod(Smearing.__init__, rbf)

class ERBFSmearing(Smearing):
    __init__ = partialmethod(Smearing.__init__, erbf)

class SpatialAttention(Block):
    """Spatial attention module. """
    def __init__(self):
        super().__init__()
        self.linear_k = Linear(activation=None, bias=False)
        self.linear_q = Linear(activation=None, bias=False)
        self.linear_summarize = Linear()
        self.linear = Linear()

    def sample(self):
        return self.linear.sample()._replace(cls=self.__class__)

    def forward(
            self, 
            v: torch.Tensor, e: torch.Tensor, x: torch.Tensor, p: torch.Tensor,
            config: Optional[NamedTuple] = None,
        ):
        """

        Examples
        --------
        >>> spatial_attention = SpatialAttention()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(2, 2, 4)
        >>> x = torch.zeros(2, 3, 6)
        >>> p = torch.zeros(2, 3, 7)
        >>> v, e, x, p = spatial_attention(v, e, x, p)
        """
        if config is None:
            config = self.sample()

        x_k = self.linear_k(p, config=config)
        x_q = self.linear_q(p, config=config)
        a = torch.linalg.norm(x_k.unsqueeze(-3) - x_q.unsqueeze(-4), dim=-2)
        a = self.linear_summarize(a, config=config)
        e = self.linear(e, config=config)
        e = a + e
        return v, e, x, p
