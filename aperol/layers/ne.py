import torch
from ..module import Module
from ..endomorphism import Endomorphism
from ..state import State

class NodeToEdgeBroadcast(Module):
    """Broadcast receiver node features to edge features.

    Adds a learned projection of destination node j's features to each edge
    (i→j). Sender features are handled separately by NodeToEdgeSenderBroadcast
    or via the initial edge construction in ProjectionIn.

    Examples
    --------
    >>> import torch
    >>> from ..test_utils import get_random_state, get_simple_endomorphism
    >>> state = get_random_state()
    >>> endomorphism = get_simple_endomorphism()
    >>> broadcast = NodeToEdgeBroadcast(endomorphism)
    >>> new_state = broadcast(state)
    >>> assert new_state.edge.shape == state.edge.shape

    """
    def __init__(self, endomorphism: Endomorphism):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()
        self.endomorphism = endomorphism

    def initialize_parameters(self, state):
        self.weight.materialize((state.node.shape[-1], state.edge.shape[-1]))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, state: State) -> State:
        node_msg = self.endomorphism(state.node @ self.weight)  # (..., N, De)
        # unsqueeze(-3): (..., 1, N, De) → broadcasts over sender dim
        new_edge = state.edge + node_msg.unsqueeze(-3)
        return state.replace(edge=new_edge)


class NodeToEdgeSenderBroadcast(Module):
    """Broadcast sender node features to edge features.

    Adds a learned projection of source node i's features to each edge
    (i→j). Complements NodeToEdgeBroadcast which handles receiver features.

    Examples
    --------
    >>> import torch
    >>> from ..test_utils import get_random_state, get_simple_endomorphism
    >>> state = get_random_state()
    >>> endomorphism = get_simple_endomorphism()
    >>> broadcast = NodeToEdgeSenderBroadcast(endomorphism)
    >>> new_state = broadcast(state)
    >>> assert new_state.edge.shape == state.edge.shape

    """
    def __init__(self, endomorphism: Endomorphism):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()
        self.endomorphism = endomorphism

    def initialize_parameters(self, state):
        self.weight.materialize((state.node.shape[-1], state.edge.shape[-1]))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, state: State) -> State:
        node_msg = self.endomorphism(state.node @ self.weight)  # (..., N, De)
        # unsqueeze(-2): (..., N, 1, De) → broadcasts over receiver dim
        new_edge = state.edge + node_msg.unsqueeze(-2)
        return state.replace(edge=new_edge)
