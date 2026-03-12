import torch
from .state import State
from . import endomorphism
MAX_NODES = 10
MAX_FEATURES = 16

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