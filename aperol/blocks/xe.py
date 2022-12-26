"""Geometry to edge modules. """
import torch
import math
from ..module import Module

def delta_x(x):
    """Compute the vector difference among geometry. """
    return x.unsqueeze(0) - x.unsqueeze(1)

def delta_x_norm(x):
    """Compute the distance among geometry. """
    delta_x = delta_x(x)
    return torch.norm(delta_x, dim=2, keepdims=True)

def cosine_cutoff(x, lower=0.0, upper=5.0):
    """Cosine cutoff. """
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
    x = x * (x < upper)
    x = x * (x > lower)
    return cutoffs

def rbf(x, num_basis=50, lower=0.0, upper=5.0):
    """Radial basis function. """
    offset = torch.linspace(lower, upper, num_basis)
    coeff = -0.5 / (offset[1] - offset[0]) ** 2
    x = x - offset
    return torch.exp(coeff * torch.pow(x, 2))

def erbf(x, num_basis=50, lower=0.0, upper=5.0):
    """Exponential radial basis funciton. """
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

# class Smearing(Module):
