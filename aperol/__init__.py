"""Aperol — equivariant neural networks for molecular force fields."""

__version__ = "0.1.0"

from .state import State
from .module import Module
from .utils import ProjectionIn, ProjectionOut
from .data.md17 import load_md17, collate_md17, MD17Sample
from .endomorphism import LazySquareLinear, LazyResidualLinear, LazyLayerNorm, LazySwiGLU, LazySelfAttention
from .endomorphism import NodeEndomorphism, EdgeEndomorphism
from .layers import (
    NodeToEdgeBroadcast, NodeToEdgeSenderBroadcast,
    EdgeToNodeAttention, EdgeToNodeMean, EdgeToNodeMax,
    EdgeToPositionAggregation, EdgeToVelocityAggregation,
    AngleToEdgeMultiChannel,
    VelocityProjection, VelocityDotToEdge, VelocityNormToNode,
    VelocityToPositionProjection, PositionToVelocityKick,
    NodeToVelocityDamping,
    PositionToEdgeERBFSmearing, PositionToEdgeRBFSmearing, PositionToEdgeSpatialAttention,
)

__all__ = [
    "State", "Module",
    "ProjectionIn", "ProjectionOut",
    "load_md17", "collate_md17", "MD17Sample",
    # Endomorphisms
    "LazySquareLinear", "LazyResidualLinear", "LazyLayerNorm", "LazySwiGLU", "LazySelfAttention",
    "NodeEndomorphism", "EdgeEndomorphism",
    # Layers
    "NodeToEdgeBroadcast", "NodeToEdgeSenderBroadcast",
    "EdgeToNodeAttention", "EdgeToNodeMean", "EdgeToNodeMax",
    "EdgeToPositionAggregation", "EdgeToVelocityAggregation",
    "AngleToEdgeMultiChannel",
    "VelocityProjection", "VelocityDotToEdge", "VelocityNormToNode",
    "VelocityToPositionProjection", "PositionToVelocityKick",
    "NodeToVelocityDamping",
    "PositionToEdgeERBFSmearing", "PositionToEdgeRBFSmearing", "PositionToEdgeSpatialAttention",
]
