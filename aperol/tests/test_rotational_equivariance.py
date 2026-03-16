import pytest
import torch

from aperol.test_utils import (
    get_random_rotation_matrix,
    get_random_state,
    get_simple_endomorphism,
    check_layer,
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

LAYER_FACTORIES = [
    ("NodeToEdgeBroadcast", lambda: NodeToEdgeBroadcast(get_simple_endomorphism())),
    ("VelocityDotToEdge", lambda: VelocityDotToEdge(get_simple_endomorphism())),
    ("VelocityNormToNode", lambda: VelocityNormToNode(get_simple_endomorphism())),
    ("VelocityProjection", lambda: VelocityProjection()),
    ("VelocityToPositionProjection", lambda: VelocityToPositionProjection()),
    ("PositionToVelocityKick", lambda: PositionToVelocityKick()),
    ("PositionToEdgeRBFSmearing", lambda: PositionToEdgeRBFSmearing()),
    ("PositionToEdgeERBFSmearing", lambda: PositionToEdgeERBFSmearing()),
    ("PositionToEdgeSpatialAttention", lambda: PositionToEdgeSpatialAttention(get_simple_endomorphism())),
    ("EdgeToNodeMean", lambda: EdgeToNodeMean(get_simple_endomorphism())),
    ("EdgeToNodeMax", lambda: EdgeToNodeMax(get_simple_endomorphism())),
    ("EdgeToNodeAttention", lambda: EdgeToNodeAttention(get_simple_endomorphism())),
    ("NodeToVelocityDamping", lambda: NodeToVelocityDamping(get_simple_endomorphism())),
]


@pytest.mark.parametrize(("name", "factory"), LAYER_FACTORIES)
def test_layers_are_rotationally_equivariant(name, factory):
    torch.manual_seed(0)
    atol, rtol = 1e-5, 1e-4
    trials = 10

    for _ in range(trials):
        state = get_random_state()
        r = get_random_rotation_matrix(device=state.position.device, dtype=state.position.dtype)
        layer = factory()
        check_layer(layer, name=name, state=state, r=r, atol=atol, rtol=rtol)

