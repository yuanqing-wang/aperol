"""Angle-to-edge layers: inject geometric angle features into edge features.

All layers are rotation-invariant: angle cosines are dot-product-based
and invariant under orthogonal transformations.
"""

import torch
from ..module import Module
from ..state import State


class AngleToEdge(Module):
    """Inject mean 3-body angle cosines (averaged over position channels) into edges.

    For directed edge (i→j): computes mean_k cos(angle at i between j and k),
    averaged over all k and over position feature channels. Adds a learned
    per-edge scaling of this invariant scalar to each edge feature.

    Examples
    --------
    >>> from ..test_utils import get_random_state
    >>> state = get_random_state()
    >>> layer = AngleToEdge()
    >>> new_state = layer(state)
    >>> assert new_state.edge.shape == state.edge.shape
    """

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()

    def initialize_parameters(self, state):
        self.weight.materialize((1, state.edge.shape[-1]))
        torch.nn.init.zeros_(self.weight)

    def forward(self, state: State) -> State:
        x = state.position.mean(dim=-1)           # (..., N, 3) — mean over Dx channels
        delta = x.unsqueeze(-2) - x.unsqueeze(-3)  # (..., N, N, 3)
        norm = torch.sqrt((delta ** 2).sum(dim=-1, keepdim=True) + 1e-12)
        unit = delta / norm                        # (..., N, N, 3)
        cos_angles = torch.einsum("...ijd,...ikd->...ijk", unit, unit)  # (..., N, N, N)
        mean_cos = cos_angles.mean(dim=-1)         # (..., N, N)
        return state.replace(edge=state.edge + mean_cos.unsqueeze(-1) * self.weight)


class AngleToEdgeMultiChannel(Module):
    """Inject per-channel 3-body angle cosines into edges.

    Like AngleToEdge but computes angles independently for each position
    feature channel (Dx channels) and projects the resulting Dx-dimensional
    angle vector to edge features via a learned (Dx × De) matrix.

    Provides richer geometric information: each position feature channel can
    capture different aspects of the local geometry.

    Examples
    --------
    >>> from ..test_utils import get_random_state
    >>> state = get_random_state()
    >>> layer = AngleToEdgeMultiChannel()
    >>> new_state = layer(state)
    >>> assert new_state.edge.shape == state.edge.shape
    """

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()

    def initialize_parameters(self, state):
        Dx = state.position.shape[-1]
        De = state.edge.shape[-1]
        self.weight.materialize((Dx, De))
        torch.nn.init.zeros_(self.weight)

    def forward(self, state: State) -> State:
        # Compute angles independently for each position channel d
        # state.position: (..., N, 3, Dx)
        x = state.position                          # (..., N, 3, Dx)
        # delta[..., i, j, :, d] = position[i, :, d] - position[j, :, d]
        # Use same unsqueeze pattern as other layers:
        # unsqueeze(-3) on (..., N, 3, Dx) → (..., N, 1, 3, Dx)
        # unsqueeze(-4) on (..., N, 3, Dx) → (..., 1, N, 3, Dx)
        delta = x.unsqueeze(-3) - x.unsqueeze(-4)  # (..., N, N, 3, Dx)
        norm = torch.sqrt((delta ** 2).sum(dim=-2, keepdim=True) + 1e-12)  # (..., N, N, 1, Dx)
        unit = delta / norm                         # (..., N, N, 3, Dx)
        # cos_angles[..., i, j, k, d] = unit[i,j,:,d] · unit[i,k,:,d]
        cos_angles = torch.einsum("...ijsd,...iksd->...ijkd", unit, unit)  # (..., N, N, N, Dx)
        mean_cos = cos_angles.mean(dim=-2)          # (..., N, N, Dx) — mean over k
        edge_update = mean_cos @ self.weight        # (..., N, N, De)
        return state.replace(edge=state.edge + edge_update)
