from dataclasses import dataclass, replace
import torch

@dataclass(frozen=True)
class State:
    node: torch.Tensor     # (..., N, Dn)
    edge: torch.Tensor     # (..., N, N, De)
    position: torch.Tensor # (..., N, 3, Dx)
    velocity: torch.Tensor # (..., N, 3, Dv)

    def replace(self, **kwargs) -> "State":
        return replace(self, **kwargs)

    def to(self, device) -> "State":
        return State(
            node=self.node.to(device),
            edge=self.edge.to(device),
            position=self.position.to(device),
            velocity=self.velocity.to(device),
        )

    def cuda(self) -> "State":
        return self.to(torch.device("cuda"))

    def detach(self) -> "State":
        return State(
            node=self.node.detach(),
            edge=self.edge.detach(),
            position=self.position.detach(),
            velocity=self.velocity.detach(),
        )
