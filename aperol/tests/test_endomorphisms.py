"""Tests for endomorphism modules."""

import pytest
import torch

from aperol.endomorphism import (
    LazyLayerNorm,
    LazySelfAttention,
    LazySquareLinear,
    LazySwiGLU,
)

ENDOMORPHISMS = [
    ("LazySquareLinear", LazySquareLinear),
    ("LazyLayerNorm", LazyLayerNorm),
    ("LazySwiGLU", LazySwiGLU),
    ("LazySelfAttention", LazySelfAttention),
]

SHAPES = [
    (5, 8),
    (3, 16),
    (1, 4),
    (10, 32),
]


@pytest.mark.parametrize(("name", "cls"), ENDOMORPHISMS)
@pytest.mark.parametrize("shape", SHAPES)
def test_endomorphism_preserves_shape(name, cls, shape):
    """Each endomorphism must output the same shape as input."""
    module = cls()
    x = torch.randn(*shape)
    y = module(x)
    assert y.shape == x.shape, f"{name}: expected shape {x.shape}, got {y.shape}"


@pytest.mark.parametrize(("name", "cls"), ENDOMORPHISMS)
def test_endomorphism_is_differentiable(name, cls):
    """Each endomorphism must support autograd."""
    module = cls()
    x = torch.randn(4, 8, requires_grad=True)
    y = module(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None, f"{name}: gradient did not flow"
    assert x.grad.shape == x.shape


@pytest.mark.parametrize(("name", "cls"), ENDOMORPHISMS)
def test_endomorphism_batched(name, cls):
    """Each endomorphism must handle batched inputs."""
    module = cls()
    x = torch.randn(2, 5, 8)  # batch of 2
    y = module(x)
    assert y.shape == x.shape, f"{name}: batched shape mismatch"
