
import torch
from functools import partialmethod
from .module import Module

class Endomorphism(Module):
    pass

class FieldEndomorphism(Endomorphism):
    def __init__(
        self,
        layers: torch.nn.Module,
        field: str = "node",
    ):
        super().__init__()
        self.layers = layers
        self.field = field
        assert self.field in ["node", "edge"], f"Unknown field: {self.field}"
        
    def forward(self, state):
        if self.field == "node":
            return state.replace(node=self.layers(state.node))
        elif self.field == "edge":
            return state.replace(edge=self.layers(state.edge))
        else:
            raise ValueError(f"Unknown field: {self.field}")
        
class NodeEndomorphism(FieldEndomorphism):
    __init__ = partialmethod(FieldEndomorphism.__init__, field="node")
        
class EdgeEndomorphism(FieldEndomorphism):
    __init__ = partialmethod(FieldEndomorphism.__init__, field="edge")
    
    
# =============================================================================
# common endomorphisms
# =============================================================================

class LazySquareLinear(Endomorphism):
    """ Lazy linear layer, where the input feature equals output.
    
    Examples
    --------
    >>> linear = LazySquareLinear()
    >>> x = torch.randn(5, 8)
    >>> y = linear(x)
    >>> assert y.shape == x.shape
    """
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()
        self.bias = torch.nn.UninitializedParameter()
        
    def initialize_parameters(self, x):
        self.weight.materialize((x.shape[-1], x.shape[-1]))
        self.bias.materialize((x.shape[-1],))
        
    def forward(self, x):
        return x @ self.weight + self.bias
    
class LazyLayerNorm(Endomorphism):
    """ Lazy layer norm.
    
    Examples
    --------
    >>> norm = LazyLayerNorm()
    >>> x = torch.randn(5, 8)
    >>> y = norm(x)
    >>> assert y.shape == x.shape
    """
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()
        self.bias = torch.nn.UninitializedParameter()
        
    def initialize_parameters(self, x):
        self.weight.materialize((x.shape[-1],))
        self.bias.materialize((x.shape[-1],))
        
    def forward(self, x):
        return torch.nn.functional.layer_norm(
            input=x,
            normalized_shape=x.shape[-1:],
            weight=self.weight,
            bias=self.bias,
        )
        
class LazySelfAttention(Endomorphism):
    """ Lazy self attention.
    
    Examples
    --------
    >>> att = LazySelfAttention()
    >>> x = torch.randn(5, 8)
    >>> y = att(x)
    >>> assert y.shape == x.shape
    """
    def __init__(self):
        super().__init__()
        self.K = torch.nn.UninitializedParameter()
        self.Q = torch.nn.UninitializedParameter()
        self.V = torch.nn.UninitializedParameter()
        
    def initialize_parameters(self, x):
        self.K.materialize((x.shape[-1], x.shape[-1]))
        self.Q.materialize((x.shape[-1], x.shape[-1]))
        self.V.materialize((x.shape[-1], x.shape[-1]))
        
    def forward(self, x):
        K = x @ self.K
        Q = x @ self.Q
        V = x @ self.V
        att = torch.einsum("...ab,...cb->...ac", Q, K) / (x.shape[-1] ** 0.5)
        att = torch.nn.functional.softmax(att, dim=-2)
        return torch.einsum("...ac,...cb->...ab", att, V)
    
    
