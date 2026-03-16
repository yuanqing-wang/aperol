import argparse

import torch

from aperol.test_utils import (
    get_random_rotation_matrix,
    get_random_state,
    get_simple_endomorphism,
    rotate_state,
    rotate_xyz,
)
from aperol.layers import (
    EdgeToNodeAttention,
    EdgeToNodeMax,
    EdgeToNodeMean,
    NodeToEdgeBroadcast,
    NodeToVelocityDamping,
    PositionToEdgeERBFSmearing,
    PositionToEdgeRBFSmearing,
    PositionToEdgeSpatialAttention,
    PositionToVelocityKick,
    VelocityDotToEdge,
    VelocityNormToNode,
    VelocityProjection,
    VelocityToPositionProjection,
)
from aperol.state import State


def _assert_allclose(a: torch.Tensor, b: torch.Tensor, *, name: str, atol: float, rtol: float):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs()
        raise AssertionError(
            f"{name} not close: max_abs={diff.max().item():.3e} "
            f"mean_abs={diff.mean().item():.3e} shape={tuple(a.shape)}"
        )


def _check_layer(
    layer: torch.nn.Module,
    *,
    name: str,
    state: State,
    r: torch.Tensor,
    atol: float,
    rtol: float,
):
    state_r = rotate_state(state, r)

    out = layer(state)
    out_r = layer(state_r)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    for _ in range(args.trials):
        state = get_random_state()
        r = get_random_rotation_matrix(device=state.position.device, dtype=state.position.dtype)
        layers = [
            ("NodeToEdgeBroadcast", NodeToEdgeBroadcast(get_simple_endomorphism())),
            ("VelocityDotToEdge", VelocityDotToEdge(get_simple_endomorphism())),
            ("VelocityNormToNode", VelocityNormToNode(get_simple_endomorphism())),
            ("VelocityProjection", VelocityProjection()),
            ("VelocityToPositionProjection", VelocityToPositionProjection()),
            ("PositionToVelocityKick", PositionToVelocityKick()),
            ("PositionToEdgeRBFSmearing", PositionToEdgeRBFSmearing()),
            ("PositionToEdgeERBFSmearing", PositionToEdgeERBFSmearing()),
            ("PositionToEdgeSpatialAttention", PositionToEdgeSpatialAttention(get_simple_endomorphism())),
            ("EdgeToNodeMean", EdgeToNodeMean(get_simple_endomorphism())),
            ("EdgeToNodeMax", EdgeToNodeMax(get_simple_endomorphism())),
            ("EdgeToNodeAttention", EdgeToNodeAttention(get_simple_endomorphism())),
            ("NodeToVelocityDamping", NodeToVelocityDamping(get_simple_endomorphism())),
        ]
        for name, layer in layers:
            _check_layer(layer, name=name, state=state, r=r, atol=args.atol, rtol=args.rtol)

    print("ok: all layers are rotationally equivariant (within tolerance)")


if __name__ == "__main__":
    main()
