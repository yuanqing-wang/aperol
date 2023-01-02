import math
import torch
from aperol.constants import MAX_IN, MAX_OUT, MIN_IN, MIN_OUT
MIN_N_NODES = 2
MAX_N_NODES = 32
BATCH_SIZE = 64

def get_inputs():
    n = torch.randint(low=MIN_N_NODES, high=MAX_N_NODES, size=()).item()
    cv = torch.randint(low=MIN_IN, high=MAX_IN, size=()).item()
    ce = torch.randint(low=MIN_IN, high=MAX_IN, size=()).item()
    cx = torch.randint(low=MIN_IN, high=MAX_IN, size=()).item()

    v = torch.randn(BATCH_SIZE, n, cv)
    e = torch.randn(BATCH_SIZE, n, n, ce)
    x = torch.randn(BATCH_SIZE, n, 3, cx)
    return v, e, x

def assert_number_of_dimensions_consistent(v0, e0, x0, v1, e1, x1):
    assert v1.dim() + 1 == e1.dim() == x1.dim()

def assert_number_of_dimensions_unchanged(v0, e0, x0, v1, e1, x1):
    assert v0.dim() == v1.dim()
    assert e0.dim() == e1.dim()
    assert x0.dim() == x1.dim()

def assert_number_of_particles_unchanged(v0, e0, x0, v1, e1, x1):
    assert v1.shape[-2] == e1.shape[-2] == e1.shape[-3] == x1.shape[-3]

def assert_dimension_reasonable(v0, e0, x0, v1, e1, x1):
    assert_number_of_dimensions_consistent(v0, e0, x0, v1, e1, x1)
    assert_number_of_dimensions_unchanged(v0, e0, x0, v1, e1, x1)
    assert_number_of_particles_unchanged(v0, e0, x0, v1, e1, x1)

def get_translation_rotation_reflection():
    x_translation = torch.distributions.Normal(
        torch.zeros(3, 1),
        torch.ones(3, 1),
    ).sample()
    translation = lambda x: x + x_translation

    import math
    alpha = torch.distributions.Uniform(-math.pi, math.pi).sample().item()
    beta = torch.distributions.Uniform(-math.pi, math.pi).sample().item()
    gamma = torch.distributions.Uniform(-math.pi, math.pi).sample().item()

    rz = torch.tensor(
        [
            [math.cos(alpha), -math.sin(alpha), 0],
            [math.sin(alpha),  math.cos(alpha), 0],
            [0,                0,               1],
        ]
    )

    ry = torch.tensor(
        [
            [math.cos(beta),   0,               math.sin(beta)],
            [0,                1,               0],
            [-math.sin(beta),  0,               math.cos(beta)],
        ]
    )

    rx = torch.tensor(
        [
            [1,                0,               0],
            [0,                math.cos(gamma), -math.sin(gamma)],
            [0,                math.sin(gamma), math.cos(gamma)],
        ]
    )

    rotation = lambda x: (x.swapaxes(-1, -2) @ rz @ ry @ rx).swapaxes(-1, -2)

    alpha = torch.distributions.Uniform(-math.pi, math.pi).sample().item()
    beta = torch.distributions.Uniform(-math.pi, math.pi).sample().item()
    gamma = torch.distributions.Uniform(-math.pi, math.pi).sample().item()
    v = torch.tensor([[alpha, beta, gamma]])
    v /= v.norm()

    p = torch.eye(3) - 2 * v.T @ v

    reflection = lambda x: (x.swapaxes(-1, -2) @ p).swapaxes(-1, -2)

    return translation, rotation, reflection
