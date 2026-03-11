from dataclasses import dataclass, replace
import torch

@dataclass(frozen=True)
class State:
    node: torch.Tensor # (N, D)
    edge: torch.Tensor # (N, N, D)
    position: torch.Tensor # (N, 3, Dx)
    velocity: torch.Tensor # (N, 3, Dv)
    
    def replace(self, **kwargs):
        return replace(self, **kwargs)
