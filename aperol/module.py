import abc
from typing import Optional, NamedTuple, Callable
from collections import namedtuple
import torch
from .constants import MAX_IN, MAX_OUT, MIN_IN, MIN_OUT, ACTIVATION

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
            Equivariant representation.

        Returns
        -------
        torch.Tensor
            Node representation.

        torch.Tensor
            Edge representation.

        torch.Tensor
            Equivariant representation.
        """
        if config is None:
            config = self.sample()

    @property
    @abc.abstractmethod
    def Config(self):
        """Factory to generate configs."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self) -> NamedTuple:
        """Sample a configuration."""
        raise NotImplementedError

class ParametrizedModule(Module):
    """Base module with weight and bias parameters."""
    def __init__(
            self,
            max_in: int = MAX_IN,
            max_out: int = MAX_OUT,
            min_in: int = MIN_IN,
            min_out: int = MIN_OUT,
            activation: Callable = ACTIVATION,
        ):
        super().__init__()
        self.max_in = max_in
        self.max_out = max_out
        self.min_in = min_in
        self.min_out = min_out
        self.activation = activation
        self.W = torch.nn.Parameter(torch.Tensor(max_in, max_out))
        self.B = torch.nn.Parameter(torch.Tensor(max_out))
        self.initialize()

    def initialize(self):
        """Initialize weight and bias. """
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.zeros_(self.B)

    @property
    def Config(self):
        """
        Examples
        --------
        >>> parametrized_module = ParametrizedModule()
        >>> config = parametrized_module.Config(5)
        >>> config.out_features
        5
        """

        return namedtuple("Config", ["out_features"])

    def sample(self):
        return self.Config(
            out_features=torch.randint(
                low=self.min_out, high=self.max_out, size=(),
            ),
        )

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
        >>> parametrized_module = ParametrizedModule()
        >>> W, B = parametrized_module.slice(2666, 1984)
        Traceback (most recent call last):
        ...
        AssertionError: assert ... <= ...

        >>> W, B = parametrized_module.slice(4, 3)
        >>> list(W.shape)
        [4, 3]
        >>> list(B.shape)
        [3]
        """
        assert in_features <= self.max_in
        assert out_features <= self.max_out
        in_idxs = torch.randint(high=self.max_in, size=(in_features,))
        out_idxs = torch.randint(high=self.max_out, size=(out_features,))
        W = self.W.index_select(0, in_idxs).index_select(1, out_idxs)
        B = self.B.index_select(0, out_idxs)
        return W, B

class Skip(Module):
    def forward(
            self,
            v: torch.Tensor,
            e: torch.Tensor,
            x: torch.Tensor,
            *args, **kwargs,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return v, e, x

class Linear(ParametrizedModule):
    def __init__(
            self,
            max_in: int = MAX_IN,
            max_out: int = MAX_OUT,
            min_in: int = MIN_IN,
            min_out: int = MIN_OUT,
            activation: Callable = ACTIVATION,
        ):
        super().__init__(
            max_in=max_in,
            max_out=max_out,
            min_in=min_in,
            min_out=min_out,
            activation=activation,
        )

    def forward(self, h, config=None):
        if config is None:
            config = self.sample()
        in_features = h.shape[-1]
        out_features = config.out_features
        W, B = self.slice(in_features, out_features)
        return self.activation(h @ W + B)
