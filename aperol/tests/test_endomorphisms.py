"""Tests for endomorphism modules."""

import pytest
import torch

from aperol.endomorphism import (
    LazyLayerNorm,
    LazyResidualLinear,
    LazySelfAttention,
    LazySquareLinear,
    LazySwiGLU,
)

ENDOMORPHISMS = [
    ("LazySquareLinear", LazySquareLinear),
    ("LazyResidualLinear", LazyResidualLinear),
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


def test_lazy_layer_norm_normalizes():
    """LazyLayerNorm should normalize input to ~zero mean, ~unit std per sample."""
    torch.manual_seed(0)
    norm = LazyLayerNorm()
    # Input with large mean and std
    x = torch.randn(4, 16) * 10 + 5
    y = norm(x)
    # Layer norm normalizes over last dim — check across all elements
    assert abs(y.mean().item()) < 0.5, f"Mean should be ~0, got {y.mean().item():.3f}"
    assert abs(y.std().item() - 1.0) < 0.3, f"Std should be ~1, got {y.std().item():.3f}"


def test_lazy_residual_linear_identity_init():
    """LazyResidualLinear initialized with W=0 should output x (identity)."""
    layer = LazyResidualLinear()
    x = torch.randn(4, 8)
    y = layer(x)
    # W=0 means x + x @ W = x + 0 = x
    torch.testing.assert_close(y, x, atol=1e-6, rtol=0,
                                msg="LazyResidualLinear should start as identity (W=0)")


def test_lazy_residual_linear_trainable():
    """After one gradient step, LazyResidualLinear should produce non-identity output."""
    torch.manual_seed(7)
    layer = LazyResidualLinear()
    optimizer = torch.optim.SGD(layer.parameters(), lr=1e-2)

    x = torch.randn(4, 8)
    # Forward + backward + step
    loss = (layer(x) ** 2).sum()  # arbitrary loss
    loss.backward()
    optimizer.step()

    # After training, W != 0, so output != x
    y_after = layer(x).detach()
    assert not torch.allclose(y_after, x, atol=1e-4), \
        "After training, LazyResidualLinear should produce non-identity output"
