from dataclasses import dataclass
import torch
from aperol.module import Module
from aperol.data.md17 import MD17Sample
from aperol.state import State


@dataclass
class Prediction:
    energy: torch.Tensor  # (B,)
    force:  torch.Tensor  # (B, n_atoms, 3)

class ProjectionIn(Module):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        position_features: int,
        velocity_features: int,
    ):
        super().__init__()
        self.node_proj = torch.nn.LazyLinear(node_features)
        self.edge_proj = torch.nn.LazyLinear(edge_features)

        self.position_features = position_features
        self.velocity_features = velocity_features

    def forward(self, sample: MD17Sample):
        node = self.node_proj(sample.atom_type)  # (n_atoms, node_features)
        edge = self.edge_proj(node)
        edge = edge.unsqueeze(-2) + edge.unsqueeze(-3)  # (n_atoms, n_atoms, edge_features)
        position = (
            sample.position
            .unsqueeze(-1)
            .repeat_interleave(self.position_features, dim=-1)
        )  # (B, n_atoms, 3, position_features)
        velocity = torch.zeros(
            *sample.position.shape, self.velocity_features,
            dtype=sample.position.dtype,
            device=sample.position.device,
        )  # (B, n_atoms, 3, velocity_features)
        return State(
            node=node,
            edge=edge,
            position=position,
            velocity=velocity,
        )
        
class ProjectionOut(Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.LazyLinear(1)

    def forward(self, state: State) -> torch.Tensor:
        energy = self.linear(state.node)  # (B, n_atoms, 1)
        energy = energy.squeeze(-1).sum(-1)  # (B,)
        return energy
    

