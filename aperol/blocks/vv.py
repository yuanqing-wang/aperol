import torch
from ..module import Module

class NodeUpdate(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, h, e, x):
        h = self.linear(x)
