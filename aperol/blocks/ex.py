"""Edge to geometry modules. """

from ..module import Module, Linear

class EdgeToGeometryUpdate(Module):
    """Update from edge to geometry. """
    def __init__(self):
        super().__init__()
        self.linear = Linear(activation=None, bias=False)

    def sample(self):
        return self.linear.sample()._replace(cls=self.__class__)

    def forward(self, v, e, x, config=None):
        config = self.linear.Config(x.shape[-1])
        delta_x = self.linear(e, config=config)
        x = delta_x.mean(0) + x
        return v, e, x
