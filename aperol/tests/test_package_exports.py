"""Test that all public symbols in aperol.__init__ are importable and usable."""

import pytest
import torch
import aperol


def test_state_importable():
    state = aperol.State(
        node=torch.randn(3, 4),
        edge=torch.randn(3, 3, 4),
        position=torch.randn(3, 3, 2),
        velocity=torch.randn(3, 3, 2),
    )
    assert isinstance(state, aperol.State)


def test_projections_importable():
    proj_in = aperol.ProjectionIn(node_features=8, edge_features=8,
                                  position_features=4, velocity_features=4)
    proj_out = aperol.ProjectionOut()
    assert proj_in is not None
    assert proj_out is not None


def test_data_importable():
    assert callable(aperol.load_md17)
    assert callable(aperol.collate_md17)
    assert aperol.MD17Sample is not None


def test_endomorphisms_importable():
    """All endomorphism classes should be importable from aperol top level."""
    for name in ["LazySquareLinear", "LazyResidualLinear", "LazyLayerNorm",
                 "LazySwiGLU", "LazySelfAttention",
                 "NodeEndomorphism", "EdgeEndomorphism"]:
        cls = getattr(aperol, name)
        x = torch.randn(4, 8)
        y = cls()(x)
        assert y.shape == x.shape, f"{name}: shape mismatch {x.shape} vs {y.shape}"


def test_layers_importable():
    """Key layers should be importable from aperol top level."""
    for name in ["NodeToEdgeBroadcast", "NodeToEdgeSenderBroadcast",
                 "VelocityProjection", "VelocityNormToNode",
                 "PositionToEdgeERBFSmearing", "AngleToEdgeMultiChannel",
                 "EdgeToPositionAggregation", "EdgeToVelocityAggregation"]:
        cls = getattr(aperol, name, None)
        assert cls is not None, f"{name} not exported from aperol"
