"""Edge modules. """
import torch
from ..module import Module, Linear, BlockModule

class EdgeUpdate(BlockModule):
    def __init__(self):
        super().__init__()
        self.linear = Linear()

    def forward(self, v, e, x, config=None):
        """
        Examples
        --------
        >>> edge_update = EdgeUpdate()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(4, 4, 8)
        >>> x = torch.zeros(2, 3, 6)
        >>> config = edge_update.Config(10)
        >>> v1, e1, x1 = edge_update(v, e, x, config=config)
        >>> list(e1.shape)
        [4, 4, 10]
        >>> assert torch.isclose(v1, v).all()
        >>> assert torch.isclose(x1, x).all()
        """
        if config is None:
            config = self.sample()
        e = self.linear(e, config=config)
        return v, e, x
