import torch

class Endomorphism(torch.nn.Module):
    def __init__(
        self,
        layers: torch.nn.Module,
    ):
        super().__init__()
        self.layers = layers
        
    def forward(self, x):
        return self.layers(x)