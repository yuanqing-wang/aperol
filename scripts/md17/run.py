import os
from pathlib import Path
import torch
import wandb
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
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test rotational equivariance
    from aperol.test_utils import check_model
    check_model(Model())

    model = Model().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run_name = Path(__file__).parent.name
    start_epoch = 0
    wandb_run_id = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model = ckpt["model"].to(device)
        optimizer = ckpt["optimizer"]
        start_epoch = ckpt["epoch"] + 1
        wandb_run_id = ckpt.get("wandb_run_id")
        print(f"Resumed from {args.checkpoint} (epoch {start_epoch})")

    if wandb_run_id:
        wandb.init(project="aperol-md17", id=wandb_run_id, resume="must")
    else:
        wandb.init(project="aperol-md17", name=run_name, config=vars(args))

    for epoch in range(start_epoch, start_epoch + args.n_epoch):
        for sample in train_loader:
            sample = sample.cuda() if device.type == "cuda" else sample
            sample.position.requires_grad_(True)

            energy = model(sample)
            force = -torch.autograd.grad(
                energy.sum(),
                sample.position,
                create_graph=True,
            )[0]
            
            energy_error = torch.nn.functional.mse_loss(energy, sample.energy)
            force_error = torch.nn.functional.mse_loss(force, sample.force)

            loss = args.energy_weight * energy_error + args.force_weight * force_error
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loss on a single (cycling) batch each train step.
        model.eval()
        try:
            val_sample = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            val_sample = next(val_iter)
        val_sample = _to_device(val_sample, device)
        val_sample.position.requires_grad_(True)
        val_energy = model(val_sample)
        val_force = -torch.autograd.grad(
            val_energy.sum(),
            val_sample.position,
            create_graph=False,
        )[0]
        val_energy_mse = torch.nn.functional.mse_loss(val_energy, val_sample.energy)
        val_force_mse = torch.nn.functional.mse_loss(val_force, val_sample.force)
        model.train()

        print(
            f"epoch {epoch:>2d} | loss {loss.item():.2f} | "
            f"energy error {energy_error.item():.2f} | force error {force_error.item():.2f} | "
            f"val_e {val_energy_mse.item():.2f} | val_f {val_force_mse.item():.2f}"
        )
        
        wandb.log({
            "epoch": epoch,
            "loss": loss.item(),
            "energy_error": energy_error.item(),
            "force_error": force_error.item(),
            "val_e": val_energy_mse.item(),
            "val_f": val_force_mse.item(),
        })

        if args.checkpoint:
            torch.save({"model": model, "optimizer": optimizer, "epoch": epoch, "wandb_run_id": wandb.run.id}, args.checkpoint)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="malonaldehyde")
    parser.add_argument("--n_tr", type=int, default=1000)
    parser.add_argument("--n_vl", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_epoch", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--node_features", type=int, default=16)
    parser.add_argument("--edge_features", type=int, default=16)
    parser.add_argument("--position_features", type=int, default=8)
    parser.add_argument("--velocity_features", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--energy_weight", type=float, default=0.01)
    parser.add_argument("--force_weight", type=float, default=0.99)
    parser.add_argument("--checkpoint", type=str, default=str(Path(__file__).parent / "checkpoint.pt"))
    args = parser.parse_args()
    run(args)
