"""Superlayer implementation. """

import torch
from .module import Module, Linear
from . import blocks
from .constants import MAX_DEPTH, MAX_IN

class SuperLayer(Module):
    """SuperLayer consisting of all possible blocks. """
    def __init__(self):
        super().__init__()
        self.all_blocks = torch.nn.ModuleDict(
            {name: block() for name, block in blocks.all_blocks.items()}
        )
        self.keys = blocks.__all__

    def sample(self):
        idx = torch.randint(high=len(self.all_blocks)-1, size=()).item()
        key = self.keys[idx]
        block = self.all_blocks[key]
        config = block.sample()
        return config

    def forward(self, v, e, x, config=None):
        """

        Examples
        --------
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(2, 2, 8)
        >>> x = torch.zeros(2, 3, 6)
        >>> layer = SuperLayer()
        >>> v, e, x = layer(v, e, x)
        """
        if config is None:
            config = self.sample()
        block = self.all_blocks[config.cls.__name__]
        e = torch.nn.functional.normalize(e, p=2, dim=-1)
        v = torch.nn.functional.normalize(v, p=2, dim=-1)
        v, e, x = block(v, e, x)
        return v, e, x

class SuperModel(Module):
    """SuperModel consisting of SuperLayer. """
    def __init__(self, in_features: int, out_features: int, depth=MAX_DEPTH):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [SuperLayer() for _ in range(depth)]
        )
        self.embedding_in = torch.nn.Linear(in_features, MAX_IN)
        self.edge_left = torch.nn.Linear(in_features, MAX_IN)
        self.edge_right = torch.nn.Linear(in_features, MAX_IN)
        self.embedding_out = Linear(max_out=out_features)

    def forward(self, v, x, config=None):
        x = x.unsqueeze(-1)
        x_aux = torch.zeros(*x.shape[:-1], MAX_IN - 1, device=x.device)
        x = torch.cat([x, x_aux], dim=-1)
        e = self.edge_left(v.unsqueeze(-2)) + self.edge_right(v.unsqueeze(-3))
        for layer in self.layers:
            v, e, x = layer(v, e, x)
        config = self.embedding_out.Config(self.embedding_out.max_out)
        v = self.embedding_out(v, config=config)
        return v, x
