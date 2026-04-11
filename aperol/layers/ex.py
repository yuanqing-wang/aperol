import torch
from ..module import Module
from ..endomorphism import Endomorphism
from ..state import State


class EdgeToPositionAggregation(Module):
    """Aggregate edge-weighted relative positions into a position update.

    For each atom i, computes a weighted average of the displacement vectors
    (x_i - x_j) over all neighbors j, where the weights come from a learned
    function of the edge features. The result is added to state.position.

    This is equivariant: displacements are 3D vectors (equivariant) and edge
    weights are invariant scalars, so the weighted sum is equivariant.

    Examples
    --------
    >>> import torch
    >>> from ..test_utils import get_random_state, get_simple_endomorphism
    >>> state = get_random_state()
    >>> endomorphism = get_simple_endomorphism()
    >>> layer = EdgeToPositionAggregation(endomorphism)
    >>> new_state = layer(state)
    >>> assert new_state.position.shape == state.position.shape
    """

    def __init__(self, endomorphism: Endomorphism):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()
        self.endomorphism = endomorphism

    def initialize_parameters(self, state):
        self.weight.materialize((state.edge.shape[-1], state.position.shape[-1]))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, state: State) -> State:
        # delta[i, j, 3, Dx] = position[i] - position[j]
        delta = state.position.unsqueeze(-3) - state.position.unsqueeze(-4)  # (..., N, N, 3, Dx)
        # w[i, j, Dx] — invariant weights from edge features
        w = self.endomorphism(state.edge @ self.weight)  # (..., N, N, Dx)
        # weighted displacement, aggregated over sources j
        update = (delta * w.unsqueeze(-2)).mean(dim=-4)   # (..., N, 3, Dx)
        return state.replace(position=state.position + update)


class EdgeToVelocityAggregation(Module):
    """Aggregate edge-weighted relative velocities into a velocity update.

    Analogous to EdgeToPositionAggregation but for velocity. For each atom i,
    computes an edge-weighted mean of (v_i - v_j) and adds it to velocity.

    Equivariant: relative velocities are 3D vectors, weights are invariant.

    Examples
    --------
    >>> import torch
    >>> from ..test_utils import get_random_state, get_simple_endomorphism
    >>> state = get_random_state()
    >>> endomorphism = get_simple_endomorphism()
    >>> layer = EdgeToVelocityAggregation(endomorphism)
    >>> new_state = layer(state)
    >>> assert new_state.velocity.shape == state.velocity.shape
    """

    def __init__(self, endomorphism: Endomorphism):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()
        self.endomorphism = endomorphism

    def initialize_parameters(self, state):
        self.weight.materialize((state.edge.shape[-1], state.velocity.shape[-1]))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, state: State) -> State:
        delta = state.velocity.unsqueeze(-3) - state.velocity.unsqueeze(-4)  # (..., N, N, 3, Dv)
        w = self.endomorphism(state.edge @ self.weight)  # (..., N, N, Dv)
        update = (delta * w.unsqueeze(-2)).mean(dim=-4)   # (..., N, 3, Dv)
        return state.replace(velocity=state.velocity + update)
