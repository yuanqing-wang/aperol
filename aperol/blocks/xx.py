import torch
from ..module import ParametrizedModule

class GeometryUpdate(ParametrizedModule):
    def __init__(self):
        super().__init__()

    def forward(self, v, e, x, config=None):
        """
        Examples
        --------
        >>> geometry_update = GeometryUpdate()
        >>> v = torch.zeros(2, 5)
        >>> e = torch.zeros(3, 4)
        >>> x = torch.zeros(2, 3, 6)
        >>> config = geometry_update.Config(10)
        >>> v1, e1, x1 = geometry_update(v, e, x, config=config)
        >>> list(x1.shape)
        [2, 3, 10]
        >>> assert torch.isclose(v1, v).all()
        >>> assert torch.isclose(e1, e).all()
        """
        if config is None:
            config = self.sample()
        in_features = x.shape[-1]
        out_features = config.out_features
        W, _ = self.slice(in_features, out_features)
        x = x @ W
        return v, e, x
