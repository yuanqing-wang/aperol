"""Superlayer implementation. """

import torch
from .module import Module
from . import blocks

class SuperLayer(Module):
    """SuperLayer consisting of all possible blocks. """
    def __init__(self):
        super().__init__()
        self.all_blocks = torch.nn.ModuleDict(
            {name: block() for name, block in blocks.all_blocks.items()}
        )
        self.keys = blocks.__all__

        # print(self.all_blocks)
        # print(self.keys)

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
        v, e, x = block(v, e, x)
        return v, e, x
