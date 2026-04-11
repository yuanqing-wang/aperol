"""Base training script for MD17 malonaldehyde.

Copy this into experiments/{n}/run.py and modify as needed.
Do NOT change n_tr or n_vl — use the validated defaults (1000/1000).
All class definitions must remain identical to the active experiment chain
so that checkpoints can be loaded across experiments.
"""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from aperol.data.md17 import load_md17, collate_md17
from aperol.utils import ProjectionIn, ProjectionOut
from aperol.module import Module
from aperol.endomorphism import (
    NodeEndomorphism,
    EdgeEndomorphism,
    LazySquareLinear,
    LazyLayerNorm,
)
from aperol.state import State
from aperol.layers import (
    NodeToEdgeBroadcast,
    NodeToVelocityDamping,
    VelocityDotToEdge,
    VelocityNormToNode,
    VelocityProjection,
    VelocityToPositionProjection,
    PositionToEdgeERBFSmearing,
    PositionToEdgeSpatialAttention,
    EdgeToNodeAttention,
    PositionToVelocityKick,
)


# ---------------------------------------------------------------------------
# Building blocks — modify these freely; keep class *names* stable
# ---------------------------------------------------------------------------

def FeedForward():
    return torch.nn.Sequential(
        LazySquareLinear(),
        LazyLayerNorm(),
        torch.nn.SiLU(),
        LazySquareLinear(),
        LazyLayerNorm(),
        torch.nn.SiLU(),
    )


class AngleToEdge(Module):
    """Inject mean 3-body angle cosines into edge features (rotation-invariant)."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.UninitializedParameter()

    def initialize_parameters(self, state):
        self.weight.materialize((1, state.edge.shape[-1]))
        torch.nn.init.zeros_(self.weight)

    def forward(self, state: State) -> State:
        x = state.position.mean(dim=-1)           # (..., N, 3)
        delta = x.unsqueeze(-2) - x.unsqueeze(-3)  # (..., N, N, 3)
        norm = torch.sqrt((delta ** 2).sum(dim=-1, keepdim=True) + 1e-12)
        unit = delta / norm
        cos_angles = torch.einsum("...ijd,...ikd->...ijk", unit, unit)
        mean_cos = cos_angles.mean(dim=-1)
        return state.replace(edge=state.edge + mean_cos.unsqueeze(-1) * self.weight)


class PairBaseline(torch.nn.Module):
    """Pair potential in learned position feature space.

    Provides a direct gradient path from energy to positions via pairwise
    distances in the message-passed position feature space.
    """

    def __init__(self, n_rbf: int = 50, hidden: int = 128):
        super().__init__()
        self.n_rbf = n_rbf
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_rbf, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        delta = position.unsqueeze(-3) - position.unsqueeze(-4)
        dist_sq = (delta ** 2).sum(dim=-2)
        n = dist_sq.shape[-3]
        eye = torch.eye(n, device=position.device, dtype=position.dtype)
        dist_sq = dist_sq * (1.0 - eye).unsqueeze(-1)
        dist_per_ch = torch.sqrt(dist_sq + 1e-12)
        mean_dist = dist_per_ch.mean(dim=-1)
        lower, upper = 0.0, 10.0
        offset = torch.linspace(lower, upper, self.n_rbf, device=position.device)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        rbf_feat = torch.exp(coeff * (mean_dist.unsqueeze(-1) - offset) ** 2)
        pair_energy = self.mlp(rbf_feat).squeeze(-1)
        return (pair_energy * (1.0 - eye)).sum(dim=(-1, -2)) * 0.5


class CartesianPairBaseline(torch.nn.Module):
    """Pair potential in raw Cartesian position space.

    Complements PairBaseline by providing a direct gradient through the
    raw atomic positions before any message passing.
    """

    def __init__(self, n_rbf: int = 50, hidden: int = 128):
        super().__init__()
        self.n_rbf = n_rbf
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_rbf, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, position_raw: torch.Tensor) -> torch.Tensor:
        delta = position_raw.unsqueeze(-2) - position_raw.unsqueeze(-3)
        dist_sq = (delta ** 2).sum(dim=-1)
        n = dist_sq.shape[-1]
        eye = torch.eye(n, device=position_raw.device, dtype=position_raw.dtype)
        dist_sq = dist_sq * (1.0 - eye)
        dist = torch.sqrt(dist_sq + 1e-12)
        lower, upper = 0.0, 10.0
        offset = torch.linspace(lower, upper, self.n_rbf, device=position_raw.device)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        rbf_feat = torch.exp(coeff * (dist.unsqueeze(-1) - offset) ** 2)
        pair_energy = self.mlp(rbf_feat).squeeze(-1)
        return (pair_energy * (1.0 - eye)).sum(dim=(-1, -2)) * 0.5


# ---------------------------------------------------------------------------
# Layer — the main message-passing building block
# ---------------------------------------------------------------------------

class Layer(Module):
    def __init__(self):
        super().__init__()
        self.node_endomorphism          = NodeEndomorphism(FeedForward())
        self.node_to_edge_broadcast     = NodeToEdgeBroadcast(FeedForward())
        self.edge_endomorphism          = EdgeEndomorphism(FeedForward())
        self.velocity_projection        = VelocityProjection()
        self.velocity_dot_to_edge       = VelocityDotToEdge(FeedForward())
        self.position_to_edge_erbf      = PositionToEdgeERBFSmearing()
        self.position_to_edge_attention = PositionToEdgeSpatialAttention(FeedForward())
        self.angle_to_edge              = AngleToEdge()
        self.edge_to_node_attention     = EdgeToNodeAttention(FeedForward())
        self.velocity_norm_to_node      = VelocityNormToNode(FeedForward())
        self.node_to_velocity_damping   = NodeToVelocityDamping(FeedForward())
        self.position_to_velocity_kick  = PositionToVelocityKick()
        self.velocity_to_position_proj  = VelocityToPositionProjection()

    def forward(self, state: State) -> State:
        node0, edge0, pos0, vel0 = state.node, state.edge, state.position, state.velocity

        state = self.node_endomorphism(state)
        state = self.node_to_edge_broadcast(state)
        state = self.edge_endomorphism(state)
        state = self.velocity_projection(state)
        state = self.velocity_dot_to_edge(state)
        state = self.position_to_edge_erbf(state)
        state = self.position_to_edge_attention(state)
        state = self.angle_to_edge(state)
        state = self.edge_to_node_attention(state)
        state = self.velocity_norm_to_node(state)
        state = self.node_to_velocity_damping(state)
        state = self.position_to_velocity_kick(state)
        state = self.velocity_to_position_proj(state)

        # Layer-level residual connections
        return state.replace(
            node=state.node + node0,
            edge=state.edge + edge0,
            position=state.position + pos0,
            velocity=state.velocity + vel0,
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Model(Module):
    def __init__(
        self,
        node_features: int = 128,
        edge_features: int = 128,
        position_features: int = 16,
        velocity_features: int = 16,
        depth: int = 5,
    ):
        super().__init__()
        self.projection_in = ProjectionIn(
            node_features=node_features,
            edge_features=edge_features,
            position_features=position_features,
            velocity_features=velocity_features,
        )
        self.layers = torch.nn.Sequential(*[Layer() for _ in range(depth)])
        self.projection_out = ProjectionOut()
        self.pair_baseline = PairBaseline()
        self.cartesian_pair_baseline = CartesianPairBaseline()

    def forward(self, sample):
        state = self.projection_in(sample)
        state = self.layers(state)
        energy = self.projection_out(state)
        energy = energy + self.pair_baseline(state.position)
        energy = energy + self.cartesian_pair_baseline(sample.position)
        return energy


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run(args):
    train, val, _ = load_md17(args.data, n_tr=args.n_tr, n_vl=args.n_vl)

    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_md17
    )
    val_loader = DataLoader(
        val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_md17
    )

    assert torch.cuda.is_available(), "CUDA not available — refusing to run on CPU"
    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})", flush=True)

    # Equivariance check (eval mode to disable stochastic ops like dropout)
    from aperol.test_utils import check_model
    _m = Model(
        node_features=args.node_features,
        edge_features=args.edge_features,
        position_features=args.position_features,
        velocity_features=args.velocity_features,
        depth=args.depth,
    )
    _m.eval()
    check_model(_m)
    print("Equivariance check passed.", flush=True)
    del _m

    model = Model(
        node_features=args.node_features,
        edge_features=args.edge_features,
        position_features=args.position_features,
        velocity_features=args.velocity_features,
        depth=args.depth,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
    )

    run_name = Path(__file__).parent.name
    start_epoch = 0
    wandb_run_id = None

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = ckpt["model"].to(device)
        optimizer = ckpt["optimizer"]
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        if "scheduler" in ckpt:
            scheduler = ckpt["scheduler"]
        start_epoch = ckpt["epoch"] + 1
        wandb_run_id = ckpt.get("wandb_run_id")
        print(f"Resumed from {args.checkpoint} at epoch {start_epoch}", flush=True)
    elif args.init_from and os.path.exists(args.init_from):
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        model = ckpt["model"].to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
        )
        print(f"Loaded model from {args.init_from}, fresh optimizer LR={args.learning_rate}", flush=True)

    if wandb_run_id:
        wandb.init(project="aperol-md17", id=wandb_run_id, resume="must")
    else:
        wandb.init(project="aperol-md17", name=run_name, config=vars(args))

    best_val_force = float("inf")

    for epoch in range(start_epoch, start_epoch + args.n_epoch):
        # ---- Training ----
        model.train()
        train_force_sum = 0.0
        train_energy_sum = 0.0
        train_n = 0

        for sample in train_loader:
            sample = sample.cuda()
            sample.position.requires_grad_(True)
            energy = model(sample)
            force = -torch.autograd.grad(
                energy.sum(), sample.position, create_graph=True
            )[0]
            energy_loss = F.mse_loss(energy, sample.energy)
            force_loss = F.mse_loss(force, sample.force)
            loss = args.energy_weight * energy_loss + args.force_weight * force_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            B = sample.energy.shape[0]
            train_energy_sum += energy_loss.item() * B
            train_force_sum += force_loss.item() * B
            train_n += B

        train_energy_error = train_energy_sum / train_n
        train_force_error = train_force_sum / train_n

        # ---- Full-epoch validation ----
        model.eval()
        val_force_sum = 0.0
        val_energy_sum = 0.0
        val_n = 0

        for val_batch in val_loader:
            val_batch = val_batch.cuda()
            val_batch.position.requires_grad_(True)
            val_energy = model(val_batch)
            val_force = -torch.autograd.grad(
                val_energy.sum(), val_batch.position, create_graph=False
            )[0]
            B = val_batch.energy.shape[0]
            val_energy_sum += F.mse_loss(val_energy, val_batch.energy).item() * B
            val_force_sum += F.mse_loss(val_force, val_batch.force).item() * B
            val_n += B

        val_energy_error = val_energy_sum / val_n
        val_force_error = val_force_sum / val_n

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"epoch {epoch:>3d} | lr {current_lr:.2e} | "
            f"train_energy {train_energy_error:.4f} | train_force {train_force_error:.4f} | "
            f"val_energy {val_energy_error:.4f} | val_force {val_force_error:.4f}",
            flush=True,
        )

        wandb.log({
            "epoch": epoch,
            "lr": current_lr,
            "train_energy_error": train_energy_error,
            "train_force_error": train_force_error,
            "val_energy_error": val_energy_error,
            "val_force_error": val_force_error,
        })

        if args.checkpoint:
            ckpt_data = {
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "epoch": epoch,
                "wandb_run_id": wandb.run.id,
            }
            torch.save(ckpt_data, args.checkpoint)
            # Also save best-val checkpoint separately for analysis/init
            if val_force_error < best_val_force:
                best_val_force = val_force_error
                best_path = Path(args.checkpoint).with_name("best_checkpoint.pt")
                torch.save(ckpt_data, best_path)
                print(f"  → new best val_force={val_force_error:.4f}, saved to {best_path.name}", flush=True)
            metrics_path = Path(args.checkpoint).with_name("metrics.jsonl")
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch,
                    "train_energy_error": round(train_energy_error, 6),
                    "train_force_error": round(train_force_error, 6),
                    "val_energy_error": round(val_energy_error, 6),
                    "val_force_error": round(val_force_error, 6),
                }) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="malonaldehyde")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_epoch", type=int, default=1)
    parser.add_argument("--n_tr", type=int, default=1000)
    parser.add_argument("--n_vl", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--node_features", type=int, default=128)
    parser.add_argument("--edge_features", type=int, default=128)
    parser.add_argument("--position_features", type=int, default=16)
    parser.add_argument("--velocity_features", type=int, default=16)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--energy_weight", type=float, default=0.01)
    parser.add_argument("--force_weight", type=float, default=0.99)
    parser.add_argument("--scheduler_step", type=int, default=10)
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--init_from", type=str, default="",
                        help="Load model weights from this checkpoint (optimizer reset)")
    parser.add_argument("--checkpoint", type=str,
                        default=str(Path(__file__).parent / "checkpoint.pt"))
    args = parser.parse_args()
    run(args)
