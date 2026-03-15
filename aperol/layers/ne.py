import torch
from ..module import Module
from ..endomorphism import Endomorphism
from ..state import State

class NodeToEdgeBroadcast(Module):
    """ Broadcast node features to edge features and add them.
    
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
        node_msg = self.endomorphism(state.node @ self.weight)  # (B, N, De)
        # broadcast receiver features across senders to match (B, N, N, De)
        node_msg = node_msg.unsqueeze(-3)
        new_edge = state.edge + node_msg
        return state.replace(edge=new_edge)
