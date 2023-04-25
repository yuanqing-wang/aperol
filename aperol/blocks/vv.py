"""Node modules. """
import torch
from ..module import Block, Linear

__all__ = ["NodeUpdate"]

class NodeUpdate(Block):
    def __init__(self):
        super().__init__()
        self.linear = Linear()

    def forward(self, v, e, x, p, config=None):
        """
        Examples
        --------
        >>> node_update = NodeUpdate()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(4, 4, 8)
        >>> x = torch.zeros(2, 3, 6)
        >>> p = torch.zeros(2, 3, 7)
        >>> config = node_update.Config(10)
        >>> v1, e1, x1, p1 = node_update(v, e, x, p, config=config)
        >>> list(v1.shape)
        [2, 10]
        >>> assert torch.isclose(e1, e).all()
        >>> assert torch.isclose(x1, x).all()
        >>> assert torch.isclose(p1, p).all()
        """
        if config is None:
            config = self.sample()
        v = self.linear(v, config=config)
        return v, e, x, p
