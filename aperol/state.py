from dataclasses import dataclass
import torch

@dataclass
class State:
    node: torch.Tensor # (N, D)
    edge: torch.Tensor # (N, N, D)
    position: torch.Tensor # (N, 3, Dx)
    velocity: torch.Tensor # (N, 3, Dv)
