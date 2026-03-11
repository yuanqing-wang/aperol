import torch

class Endomorphism(torch.nn.Module):
    def __init__(
        self,
        layers: torch.nn.Module,
        field: str = "node",
    ):
        super().__init__()
        self.layers = layers
        self.field = field
        assert self.field in ["node", "edge"], f"Unknown field: {self.field}"
        
    def forward(self, state):
        if self.field == "node":
            state.node = self.layers(state.node)
        elif self.field == "edge":
            state.edge = self.layers(state.edge)
        else:
            raise ValueError(f"Unknown field: {self.field}")
        return state
        