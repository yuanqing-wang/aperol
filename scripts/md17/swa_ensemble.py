#!/usr/bin/env python3
"""Average weights from multiple experiment checkpoints (SWA-style ensemble).

Loads checkpoints from multiple experiments, averages their state dicts,
evaluates the averaged model on the validation set, and optionally saves it
as a new checkpoint for further optimizer resets.

Why this helps: each optimizer reset finds a different local minimum.
Averaging in weight space typically yields a wider, flatter minimum that
generalizes better than any individual model.

Usage:
    python3 swa_ensemble.py 8 10 11 12         # average exps 8, 10, 11, 12
    python3 swa_ensemble.py 8 10 11 12 --save  # save as avg_checkpoint.pt
    python3 swa_ensemble.py 8 10 11 12 --save --exp 14  # save as exp14 init

Arguments:
    exp_nums          Experiment numbers to average (space-separated)
    --save            Save the averaged model checkpoint
    --exp N           Save as experiments/{N}/checkpoint.pt for use as init_from
    --data malonaldehyde  Molecule name (default: malonaldehyde)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REMOTE_BASE = "/scratch/yqw/aperol"
sys.path.insert(0, REMOTE_BASE)

from aperol.data.md17 import load_md17, collate_md17


def _import_model_class(exp_n: int):
    """Execute an experiment's run.py to bring its class definitions into scope.

    This is required because checkpoints were saved with torch.save(model) where
    model's class (Model, Layer, etc.) is defined in the experiment's run.py.
    Pickle needs the same class definitions available at load time.
    """
    import importlib.util
    run_py = Path(REMOTE_BASE) / "scripts/md17/experiments" / str(exp_n) / "run.py"
    spec = importlib.util.spec_from_file_location(f"exp{exp_n}_run", run_py)
    mod = importlib.util.module_from_spec(spec)
    # Register as __main__ so pickle finds classes there
    sys.modules["__main__"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_checkpoint(exp_n: int, prefer_best: bool = True) -> dict:
    """Load checkpoint from experiments/{n}/."""
    exp_dir = Path(REMOTE_BASE) / "scripts/md17/experiments" / str(exp_n)
    best_path = exp_dir / "best_checkpoint.pt"
    ckpt_path = exp_dir / "checkpoint.pt"

    path = best_path if (prefer_best and best_path.exists()) else ckpt_path
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found in {exp_dir}")

    print(f"  Loading exp{exp_n} from {path.name}", flush=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def average_checkpoints(ckpts: list) -> dict:
    """Average state dicts from multiple checkpoints."""
    ref_model = ckpts[0]["model"]
    ref_sd = {k: v.clone().float() for k, v in ref_model.state_dict().items()}

    for ckpt in ckpts[1:]:
        sd = ckpt["model"].state_dict()
        for k in ref_sd:
            ref_sd[k] += sd[k].float()

    n = len(ckpts)
    for k in ref_sd:
        ref_sd[k] /= n

    # Restore original dtype (usually float32, but be safe)
    orig_sd = ref_model.state_dict()
    for k in ref_sd:
        ref_sd[k] = ref_sd[k].to(orig_sd[k].dtype)

    ref_model.load_state_dict(ref_sd)
    return ref_model


def evaluate(model, data: str = "malonaldehyde", n_tr: int = 1000, n_vl: int = 1000,
             batch_size: int = 8) -> tuple[float, float]:
    """Evaluate on the full validation set. Returns (val_force_mse, val_energy_mse).

    Works on both GPU (fast) and CPU (slower but fine for 1000-sample eval).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Evaluating on {device} ...", flush=True)
    model = model.to(device)
    model.eval()

    _, val, _ = load_md17(data, n_tr=n_tr, n_vl=n_vl)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=collate_md17)

    val_force_sum = val_energy_sum = 0.0
    val_n = 0
    for batch in val_loader:
        batch = batch.to(device)
        batch.position.requires_grad_(True)
        energy = model(batch)
        force = -torch.autograd.grad(energy.sum(), batch.position, create_graph=False)[0]
        B = batch.energy.shape[0]
        val_energy_sum += F.mse_loss(energy, batch.energy).item() * B
        val_force_sum += F.mse_loss(force, batch.force).item() * B
        val_n += B

    return val_force_sum / val_n, val_energy_sum / val_n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exp_nums", type=int, nargs="+", help="Experiment numbers to average")
    parser.add_argument("--save", action="store_true", help="Save the averaged checkpoint")
    parser.add_argument("--exp", type=int, default=None,
                        help="Save as experiments/{exp}/checkpoint.pt (also creates the directory)")
    parser.add_argument("--data", type=str, default="malonaldehyde")
    parser.add_argument("--prefer-final", action="store_true",
                        help="Use checkpoint.pt instead of best_checkpoint.pt")
    args = parser.parse_args()

    print(f"SWA ensemble: averaging exps {args.exp_nums}", flush=True)
    prefer_best = not args.prefer_final

    # Warn if experiments span different chains (A/B/B+)
    # Add this script's directory to sys.path so summarize.py is importable
    # regardless of the working directory.
    _script_dir = str(Path(__file__).parent)
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    from summarize import _detect_chain
    exp_dir = str(Path(REMOTE_BASE) / "scripts/md17/experiments")
    chains = {n: _detect_chain(exp_dir, str(n)) for n in args.exp_nums}
    unique_chains = set(chains.values())
    if len(unique_chains) > 1:
        print(f"  WARNING: mixing chains {unique_chains} — averaging across different architectures "
              f"may give garbage. Continue with caution.", flush=True)
        print(f"  Chain labels: {chains}", flush=True)

    # Load Model/Layer class definitions from the first experiment's run.py.
    # All experiments in the same chain share identical class structure.
    ref_exp = args.exp_nums[0]
    print(f"  Importing class definitions from exp{ref_exp}/run.py ...", flush=True)
    _import_model_class(ref_exp)

    # Load all checkpoints
    ckpts = []
    for n in args.exp_nums:
        try:
            ckpts.append(load_checkpoint(n, prefer_best=prefer_best))
        except FileNotFoundError as e:
            print(f"  WARNING: {e} — skipping", flush=True)

    if len(ckpts) < 2:
        print("Need at least 2 checkpoints to average. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Averaging {len(ckpts)} checkpoints ...", flush=True)
    averaged_model = average_checkpoints(ckpts)

    print("Evaluating averaged model on validation set ...", flush=True)
    val_force, val_energy = evaluate(averaged_model, data=args.data)
    print(f"  val_force_mse = {val_force:.4f}  val_energy_mse = {val_energy:.4f}", flush=True)

    # Also show best individual for comparison
    best_individual = min(args.exp_nums, key=lambda n: _get_best_val(n))
    best_val = _get_best_val(best_individual)
    print(f"  Best individual: exp{best_individual} = {best_val:.4f}", flush=True)
    if val_force < best_val:
        print(f"  → IMPROVEMENT: ensemble ({val_force:.4f}) < best individual ({best_val:.4f})", flush=True)
    else:
        print(f"  → No improvement vs best individual ({best_val:.4f})", flush=True)

    if args.save:
        if args.exp is not None:
            save_dir = Path(REMOTE_BASE) / "scripts/md17/experiments" / str(args.exp)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "checkpoint.pt"
        else:
            save_path = Path(REMOTE_BASE) / "scripts/md17/experiments" / \
                f"avg_{'_'.join(str(n) for n in args.exp_nums)}" / "checkpoint.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)

        # Use the reference checkpoint structure but with averaged model
        save_data = {
            "model": averaged_model,
            "optimizer": None,   # fresh optimizer needed
            "scheduler": None,   # fresh scheduler needed
            "epoch": -1,
            "wandb_run_id": None,
            "swa_source_exps": args.exp_nums,
            "swa_val_force": val_force,
        }
        torch.save(save_data, save_path)
        print(f"  Saved to {save_path}", flush=True)


def _get_best_val(exp_n: int) -> float:
    m = Path(REMOTE_BASE) / "scripts/md17/experiments" / str(exp_n) / "metrics.jsonl"
    if not m.exists():
        return float("inf")
    lines = [json.loads(l) for l in open(m)]
    if not lines:
        return float("inf")
    return min(x["val_force_error"] for x in lines)


if __name__ == "__main__":
    main()
