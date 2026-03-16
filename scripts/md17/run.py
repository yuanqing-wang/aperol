import torch
from torch.utils.data import DataLoader
from aperol.data.md17 import load_md17, collate_md17
from aperol.utils import ProjectionIn, ProjectionOut
from aperol.module import Module
from aperol.endomorphism import (
    Endomorphism,
    NodeEndomorphism,
    EdgeEndomorphism,
    LazySquareLinear,
    LazyLayerNorm,
)

from aperol.state import State
from aperol.layers import (
    EdgeToNodeAggregation, EdgeToNodeMean, EdgeToNodeMax, AttentionAggregation, EdgeToNodeAttention,
    NodeToEdgeBroadcast,
    NodeToVelocityDamping,
    VelocityDotToEdge,
    VelocityNormToNode,
    VelocityProjection,
    VelocityToPositionProjection,
    PositionToEdgeRBFSmearing, PositionToEdgeERBFSmearing, PositionToEdgeSpatialAttention,
    PositionToVelocityKick,
)

FeedForward = lambda: torch.nn.Sequential(
        LazySquareLinear(),
        LazyLayerNorm(),
        torch.nn.SiLU(),
        LazySquareLinear(),
        LazyLayerNorm(),
        torch.nn.Tanh(),
    )

def run(args):
    train, val, _ = load_md17(
        args.data,
        n_tr=args.n_tr,
        n_vl=args.n_vl,
    )

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_md17)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_md17)
    val_iter = iter(val_loader)
    
    class Layer(Module):
        def __init__(self, FeedForward: type):
            super().__init__()
            self.node_endomorphism                = NodeEndomorphism(FeedForward())
            self.node_to_edge_broadcast           = NodeToEdgeBroadcast(FeedForward())
            self.edge_endomorphism                = EdgeEndomorphism(FeedForward())
            self.velocity_projection              = VelocityProjection()
            self.velocity_dot_to_edge             = VelocityDotToEdge(FeedForward())
            self.position_to_edge_erbf_smearing   = PositionToEdgeERBFSmearing()
            self.position_to_edge_spatial_attention = PositionToEdgeSpatialAttention(FeedForward())
            self.edge_to_node_attention           = EdgeToNodeAttention(FeedForward())
            self.node_to_velocity_damping         = NodeToVelocityDamping(FeedForward())
            self.position_to_velocity_kick        = PositionToVelocityKick()
            self.velocity_to_position_projection  = VelocityToPositionProjection()


        def forward(self, state: State) -> State:
            state = self.node_endomorphism(state)
            state = self.node_to_edge_broadcast(state)
            state = self.edge_endomorphism(state)
            state = self.velocity_projection(state)
            state = self.velocity_dot_to_edge(state)
            state = self.position_to_edge_erbf_smearing(state)
            state = self.position_to_edge_spatial_attention(state)
            state = self.edge_to_node_attention(state)
            state = self.node_to_velocity_damping(state)
            state = self.position_to_velocity_kick(state)
            state = self.velocity_to_position_projection(state)
            return state
    
    
    class Model(Module):
        def __init__(
            self,
            node_features: int = args.node_features,
            edge_features: int = args.edge_features,
            position_features: int = args.position_features,
            velocity_features: int = args.velocity_features,
            depth: int = args.depth,
        ):
            super().__init__()
            self.projection_in = ProjectionIn(
                node_features=node_features,
                edge_features=edge_features,
                position_features=position_features,
                velocity_features=velocity_features,
            )
            
            self.layers = torch.nn.Sequential(*[Layer(FeedForward) for _ in range(depth)])
            self.projection_out = ProjectionOut()

        def forward(self, sample):
            state = self.projection_in(sample)
            state = self.layers(state)
            energy = self.projection_out(state)
            return energy
        
    class Loss(torch.nn.Module):
        def __init__(
            self,
            energy_weight: float = args.energy_weight,
            force_weight: float = args.force_weight,
        ):
            super().__init__()
            self.energy_weight = energy_weight
            self.force_weight = force_weight

        def forward(
            self,
            energy_predicted: torch.Tensor,
            force_predicted: torch.Tensor,
            sample: torch.Tensor,
        ):
            energy_true = sample.energy
            force_true = sample.force
            energy_loss = torch.nn.functional.mse_loss(energy_predicted, energy_true)
            force_loss = torch.nn.functional.mse_loss(force_predicted, force_true)
            return self.energy_weight * energy_loss + self.force_weight * force_loss
        
    model = Model()
    loss_fn = Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.n_epoch):
        for sample in train_loader:
            sample.position.requires_grad_(True)

            energy = model(sample)
            force = -torch.autograd.grad(
                energy.sum(),
                sample.position,
                create_graph=True,
            )[0]

            l = loss_fn(energy, force, sample)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            # Validation loss on a single (cycling) batch each train step.
            model.eval()
            try:
                val_sample = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_sample = next(val_iter)
            val_sample.position.requires_grad_(True)
            val_energy = model(val_sample)
            val_force = -torch.autograd.grad(
                val_energy.sum(),
                val_sample.position,
                create_graph=False,
            )[0]
            val_l = loss_fn(val_energy, val_force, val_sample)
            model.train()

            print(f"epoch {epoch:>6d} | loss {l.item():.6f} | val {val_l.item():.6f}")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="malonaldehyde")
    parser.add_argument("--n_tr", type=int, default=1000)
    parser.add_argument("--n_vl", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_epoch", type=int, default=100000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--node_features", type=int, default=16)
    parser.add_argument("--edge_features", type=int, default=16)
    parser.add_argument("--position_features", type=int, default=8)
    parser.add_argument("--velocity_features", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--energy_weight", type=float, default=0.01)
    parser.add_argument("--force_weight", type=float, default=0.99)
    args = parser.parse_args()
    run(args)
