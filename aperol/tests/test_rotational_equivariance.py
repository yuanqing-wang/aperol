import pytest
import torch

from aperol.test_utils import (
    get_random_rotation_matrix,
    get_random_sample,
    get_random_state,
    get_simple_endomorphism,
    check_layer,
    check_model,
)
from aperol.utils import ProjectionIn, ProjectionOut
from aperol.module import Module
from aperol.endomorphism import NodeEndomorphism, EdgeEndomorphism, LazySquareLinear, LazyLayerNorm
from aperol.layers import EdgeToNodeMean, PositionToEdgeERBFSmearing
from aperol.state import State
from aperol.layers import (
    AngleToEdgeMultiChannel,
    EdgeToNodeAttention,
    EdgeToNodeMax,
    EdgeToNodeMean,
    EdgeToPositionAggregation,
    EdgeToVelocityAggregation,
    NodeToEdgeBroadcast,
    NodeToEdgeSenderBroadcast,
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
    ("NodeToEdgeSenderBroadcast", lambda: NodeToEdgeSenderBroadcast(get_simple_endomorphism())),
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
    ("EdgeToPositionAggregation", lambda: EdgeToPositionAggregation(get_simple_endomorphism())),
    ("EdgeToVelocityAggregation", lambda: EdgeToVelocityAggregation(get_simple_endomorphism())),
    ("AngleToEdgeMultiChannel", lambda: AngleToEdgeMultiChannel()),
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


def _make_simple_model():
    """Minimal Model for testing that connects positions to energy.

    The chain position→edge→node→energy is needed so forces can be computed
    via autograd. Uses: ProjectionIn → PositionToEdgeERBFSmearing → EdgeToNodeMean
    → ProjectionOut.
    """

    def FeedForward():
        return torch.nn.Sequential(LazySquareLinear(), LazyLayerNorm(), torch.nn.SiLU())

    class SimpleLayer(Module):
        def __init__(self):
            super().__init__()
            self.pos_to_edge = PositionToEdgeERBFSmearing()
            self.edge_to_node = EdgeToNodeMean(FeedForward())

        def forward(self, state: State) -> State:
            state = self.pos_to_edge(state)
            state = self.edge_to_node(state)
            return state

    class SimpleModel(Module):
        def __init__(self):
            super().__init__()
            self.projection_in = ProjectionIn(
                node_features=8, edge_features=8, position_features=4, velocity_features=4
            )
            self.layer = SimpleLayer()
            self.projection_out = ProjectionOut()

        def forward(self, sample):
            state = self.projection_in(sample)
            state = self.layer(state)
            return self.projection_out(state)

    return SimpleModel()


def test_model_energy_invariant_and_force_equivariant():
    """Energy E(R·x) = E(x) and force F(R·x) = R·F(x)."""
    torch.manual_seed(42)
    model = _make_simple_model()
    model.eval()
    for _ in range(5):
        sample = get_random_sample()
        r = get_random_rotation_matrix()
        check_model(model, sample=sample, r=r, atol=1e-3, rtol=1e-3)


def test_check_model_restores_training_mode():
    """check_model should restore the model's original training mode."""
    model = _make_simple_model()

    # Start in train mode — check_model should put it back
    model.train()
    assert model.training
    check_model(model)
    assert model.training, "check_model should restore train mode"

    # Start in eval mode — check_model should keep it
    model.eval()
    assert not model.training
    check_model(model)
    assert not model.training, "check_model should preserve eval mode"

