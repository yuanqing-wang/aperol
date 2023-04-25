import abc
from typing import Optional, NamedTuple, Callable
from collections import namedtuple
import torch
from .constants import MAX_IN, MAX_OUT, MIN_IN, MIN_OUT, ACTIVATION

def config(self, fields):
    return namedtuple(
        self.__class__.__name__,
        fields + ["cls"],
        defaults=(None,) * len(fields) + (self.__class__,),
    )

class Module(torch.nn.Module):
    """Base module for `aperol` building blocks. """
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abc.abstractmethod
    def forward(
            self,
            v: torch.Tensor,
            e: torch.Tensor,
            x: torch.Tensor,
            p: torch.Tensor,
            config: Optional[torch.Tensor] = None
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward call with optional config.
        If no config were specified, a random config would be sampled.

        Parameters
        ----------
        v : torch.Tensor
            Node representation.

        e : torch.Tensor
            Edge representation.

        x : torch.Tensor
            Geometry representation.

        p : torch.Tensor
            Momentum representation.

        Returns
        -------
        torch.Tensor
            Node representation.

        torch.Tensor
            Edge representation.

        torch.Tensor
            Geometry representation.

        torch.Tensor
            Momentum representation.
        """
        if config is None:
            config = self.sample()

    @property
    def Config(self):
        """
        Examples
        --------
        >>> parametrized_module = Module()
        >>> config = parametrized_module.Config(5)
        >>> config.out_features
        5
        """

        return config(self, ["out_features"])

    def sample(self):
        return self.Config(
            out_features=torch.randint(
                low=MIN_OUT, high=MAX_OUT, size=(),
            ),
        )

class Linear(Module):
    def __init__(
            self,
            max_in: int = MAX_IN,
            max_out: int = MAX_OUT,
            min_in: int = MIN_IN,
            min_out: int = MIN_OUT,
            activation: Optional[Callable] = ACTIVATION,
            bias: bool = True,
        ):
        super().__init__()
        if activation is None:
            activation = lambda x: x
        self.max_in = max_in
        self.max_out = max_out
        self.min_in = min_in
        self.min_out = min_out
        self.activation = activation
        self.bias = bias
        self.W = torch.nn.Parameter(torch.Tensor(max_in, max_out))
        if bias:
            self.B = torch.nn.Parameter(torch.Tensor(max_out))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weight and bias. """
        torch.nn.init.xavier_uniform_(self.W, gain=0.01)
        if self.bias:
            torch.nn.init.zeros_(self.B)

    def slice(
            self, in_features: int, out_features: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice the weight and bias parameters to desired dimensions.

        Parameters
        ----------
        in_features : int
            Input features.

        out_features : int
            Output features.

        Examples
        --------
        >>> linear = Linear()
        >>> W, B = linear.slice(4, 3)
        >>> list(W.shape)
        [4, 3]
        >>> list(B.shape)
        [3]
        """
        assert in_features <= self.max_in
        assert out_features <= self.max_out
        W = self.W[:in_features, :out_features]
        if self.bias:
            B = self.B[:out_features]
        else:
            B = 0.0
        return W, B

    def forward(self, h, config=None):
        if config is None:
            config = self.sample()
        in_features = h.shape[-1]
        out_features = config.out_features
        W, B = self.slice(in_features, out_features)
        return self.activation(h @ W + B)

class Block(Module):
    """Final modules used as building blocks. """

    serving = False
    def __init__(self):
        super().__init__()

    def serve(self, config):
        raise NotImplementedError
