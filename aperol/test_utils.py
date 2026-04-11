from typing import Optional

import torch
from .state import State
from .data.md17 import MD17Sample
from . import endomorphism
MAX_NODES = 10
MAX_FEATURES = 16

def get_random_rotation_matrix(*, device=None, dtype=None) -> torch.Tensor:
    """Generate a random 3x3 rotation matrix (SO(3))."""
    device = device if device is not None else torch.device("cpu")
    dtype = dtype if dtype is not None else torch.float32
    a = torch.randn(3, 3, device=device, dtype=dtype)
    q, r = torch.linalg.qr(a)
    d = torch.sign(torch.diag(r))
    q = q * d
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q

def rotate_xyz(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Rotate the xyz axis at dim=-2 by rotation matrix r (3x3)."""
    if x.shape[-2] != 3:
        raise ValueError(f"Expected xyz at dim=-2, got shape {tuple(x.shape)}")
    prefix = x.shape[:-2]
    tail = x.shape[-1:]
    x_flat = x.reshape(*prefix, 3, -1)
    y_flat = torch.einsum("ij,...jk->...ik", r, x_flat)
    return y_flat.reshape(*prefix, 3, *tail)

def rotate_state(state: State, r: torch.Tensor) -> State:
    """Rotate position and velocity in a State by r (3x3)."""
    return state.replace(
        position=rotate_xyz(state.position, r),
        velocity=rotate_xyz(state.velocity, r),
    )

def get_random_state():
    N = torch.randint(1, MAX_NODES + 1, (1,)).item()
    Dn = torch.randint(1, MAX_FEATURES + 1, (1,)).item()
    De = torch.randint(1, MAX_FEATURES + 1, (1,)).item()
    Dx = torch.randint(1, MAX_FEATURES + 1, (1,)).item()
    Dv = torch.randint(1, MAX_FEATURES + 1, (1,)).item()
    state = State(
        node=torch.randn(N, Dn),
        edge=torch.randn(N, N, De),
        position=torch.randn(N, 3, Dx),
        velocity=torch.randn(N, 3, Dv),
    )
    return state

def get_simple_endomorphism():
    return torch.nn.Sequential(
        endomorphism.LazySquareLinear(),
        endomorphism.LazyLayerNorm(),
        torch.nn.SiLU(),
        endomorphism.LazySquareLinear(),
        endomorphism.LazyLayerNorm(),
        torch.nn.SiLU(),
    )


def _assert_allclose(a: torch.Tensor, b: torch.Tensor, *, name: str, atol: float, rtol: float):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs()
        raise AssertionError(
            f"{name} not close: max_abs={diff.max().item():.3e} "
            f"mean_abs={diff.mean().item():.3e} shape={tuple(a.shape)}"
        )


def get_random_sample(n_atoms: int = 5, n_atom_types: int = 4) -> MD17Sample:
    return MD17Sample(
        position=torch.randn(n_atoms, 3),
        energy=torch.randn(1).squeeze(),
        force=torch.randn(n_atoms, 3),
        atom_type=torch.nn.functional.one_hot(
            torch.randint(n_atom_types, (n_atoms,)), n_atom_types
        ).float(),
    )


def rotate_sample(sample: MD17Sample, r: torch.Tensor) -> MD17Sample:
    """Rotate position (and force) in an MD17Sample by r (3x3)."""
    return MD17Sample(
        position=sample.position @ r.T,
        energy=sample.energy,
        force=sample.force @ r.T,
        atom_type=sample.atom_type,
    )


def check_model(
    model: torch.nn.Module,
    *,
    name: str = "Model",
    sample: Optional[MD17Sample] = None,
    r: Optional[torch.Tensor] = None,
    atol: float = 1e-3,
    rtol: float = 1e-3,
):
    if r is None:
        r = get_random_rotation_matrix()
    if sample is None:
        sample = get_random_sample()
    sample_r = rotate_sample(sample, r)

    # Switch to eval mode so stochastic layers (e.g. dropout) are deterministic.
    # Restore the original training mode when done.
    was_training = model.training
    model.eval()

    try:
        # Energy invariance check
        with torch.no_grad():
            energy = model(sample)
            energy_r = model(sample_r)
        _assert_allclose(energy, energy_r, name=f"{name}.energy", atol=atol, rtol=rtol)

        # Force equivariance check: F(R·x) = R·F(x)
        # Forces are -grad(energy, position); equivariance follows from energy
        # invariance but checking it explicitly catches gradient-path bugs.
        # Use enable_grad so this works even when called inside no_grad contexts.
        with torch.enable_grad():
            pos = sample.position.detach().requires_grad_(True)
            pos_r = sample_r.position.detach().requires_grad_(True)
            s = MD17Sample(position=pos, energy=sample.energy, force=sample.force, atom_type=sample.atom_type)
            s_r = MD17Sample(position=pos_r, energy=sample_r.energy, force=sample_r.force, atom_type=sample_r.atom_type)
            e = model(s)
            e_r = model(s_r)
            # allow_unused=True handles models whose energy doesn't depend on position
            # (e.g. pure node/edge models). If unused, forces are zero — trivially equivariant.
            _force = torch.autograd.grad(e.sum(), pos, create_graph=False, allow_unused=True)[0]
            _force_r = torch.autograd.grad(e_r.sum(), pos_r, create_graph=False, allow_unused=True)[0]
        if _force is None:
            return  # position not in graph — force equivariance trivially satisfied
        force, force_r = -_force, -_force_r
        # Positions rotate as x_r = x @ r.T; forces inherit the same transformation.
        _assert_allclose(force_r, force @ r.T, name=f"{name}.force", atol=atol, rtol=rtol)
    finally:
        if was_training:
            model.train()


def check_layer(
    layer: torch.nn.Module,
    *,
    name: str = "Layer",
    state: Optional[State] = None,
    r: Optional[torch.Tensor] = None,
    atol: float = 1e-3,
    rtol: float = 1e-3,
):
    if r is None:
        r = get_random_rotation_matrix()
    if state is None:
        state = get_random_state()
    state_r = rotate_state(state, r)

    was_training = layer.training
    layer.eval()
    try:
        out = layer(state)
        out_r = layer(state_r)
    finally:
        if was_training:
            layer.train()

    # Scalars/features should be rotation-invariant.
    _assert_allclose(out.node, out_r.node, name=f"{name}.node", atol=atol, rtol=rtol)
    _assert_allclose(out.edge, out_r.edge, name=f"{name}.edge", atol=atol, rtol=rtol)

    # Vectors should be equivariant.
    _assert_allclose(
        out_r.position,
        rotate_xyz(out.position, r),
        name=f"{name}.position",
        atol=atol,
        rtol=rtol,
    )
    _assert_allclose(
        out_r.velocity,
        rotate_xyz(out.velocity, r),
        name=f"{name}.velocity",
        atol=atol,
        rtol=rtol,
    )
