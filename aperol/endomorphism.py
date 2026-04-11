
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
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)
        
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
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)
        
    def forward(self, x):
        return torch.nn.functional.layer_norm(
            input=x,
            normalized_shape=x.shape[-1:],
            weight=self.weight,
            bias=self.bias,
        )
        
class LazySwiGLU(Endomorphism):
    """Lazy SwiGLU gated linear unit: output = SiLU(W_gate·x) * (W_proj·x).

    More expressive than a plain linear layer — the gate selectively
    amplifies or suppresses features. Same output dimension as input.
    Double the parameters of LazySquareLinear.

    Examples
    --------
    >>> gate = LazySwiGLU()
    >>> x = torch.randn(5, 8)
    >>> y = gate(x)
    >>> assert y.shape == x.shape
    """

    def __init__(self):
        super().__init__()
        self.weight_proj = torch.nn.UninitializedParameter()
        self.weight_gate = torch.nn.UninitializedParameter()
        self.bias_proj   = torch.nn.UninitializedParameter()
        self.bias_gate   = torch.nn.UninitializedParameter()

    def initialize_parameters(self, x):
        D = x.shape[-1]
        self.weight_proj.materialize((D, D))
        self.weight_gate.materialize((D, D))
        self.bias_proj.materialize((D,))
        self.bias_gate.materialize((D,))
        torch.nn.init.xavier_uniform_(self.weight_proj)
        torch.nn.init.xavier_uniform_(self.weight_gate)
        torch.nn.init.zeros_(self.bias_proj)
        torch.nn.init.zeros_(self.bias_gate)

    def forward(self, x):
        proj = x @ self.weight_proj + self.bias_proj
        gate = x @ self.weight_gate + self.bias_gate
        return torch.nn.functional.silu(gate) * proj


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
        torch.nn.init.xavier_uniform_(self.K)
        torch.nn.init.xavier_uniform_(self.Q)
        torch.nn.init.xavier_uniform_(self.V)
        
    def forward(self, x):
        K = x @ self.K
        Q = x @ self.Q
        V = x @ self.V
        att = torch.einsum("...ab,...cb->...ac", Q, K) / (x.shape[-1] ** 0.5)
        att = torch.nn.functional.softmax(att, dim=-2)
        return torch.einsum("...ac,...cb->...ab", att, V)
    
    
