import abc
import torch
from .constants import ACTIVATION


class Module(torch.nn.Module):
    """Base module for `aperol` building blocks."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    @abc.abstractmethod
    def forward(
        self,
        v: torch.Tensor,
        e: torch.Tensor,
        x: torch.Tensor,
        p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        v : torch.Tensor
            Node representation. Shape: (N, D)

        e : torch.Tensor
            Edge representation. Shape: (N, N, D)

        x : torch.Tensor
            Geometry representation. Shape: (N, 3)

        p : torch.Tensor
            Momentum representation. Shape: (N, 3, C)

        Returns
        -------
        torch.Tensor
            Node representation. Shape: (N, D)

        torch.Tensor
            Edge representation. Shape: (N, N, D)

        torch.Tensor
            Geometry representation. Shape: (N, 3)

        torch.Tensor
            Momentum representation. Shape: (N, 3, C)
        """
        raise NotImplementedError


class Linear(torch.nn.Module):
    """Lazy linear layer with optional activation.

    When `max_out` is provided it fixes the output dimension. Otherwise the
    output dimension follows the input feature size on the first call.
    """

    def __init__(
        self,
        *,
        activation=ACTIVATION,
        bias: bool | None = True,
        max_in: int | None = None,
        max_out: int | None = None,
        min_out: int | None = None,
    ):
        super().__init__()
        self.activation = activation if activation is not None else torch.nn.Identity()
        self.bias = False if bias is None else bias
        self.max_out = max_out
        self.max_in = max_in
        self.min_out = min_out
        self.linear: torch.nn.Linear | None = None

    def _init_layer(self, in_features: int, out_features: int):
        self.linear = torch.nn.Linear(in_features, out_features, bias=self.bias)

    def forward(self, x: torch.Tensor, out_features: int | None = None) -> torch.Tensor:
        in_features = x.shape[-1]
        target_out = out_features or self.max_out or in_features
        if self.max_in is not None:
            in_features = min(in_features, self.max_in)
            x = x[..., :in_features]

        if self.linear is None or \
                self.linear.in_features != in_features or \
                self.linear.out_features != target_out:
            self._init_layer(in_features, target_out)

        y = self.linear(x)
        return self.activation(y)
