#!/usr/bin/env python3
"""Evaluate a single saved checkpoint on the MD17 validation set.

Usage:
    python3 eval_checkpoint.py experiments/14/best_checkpoint.pt
    python3 eval_checkpoint.py experiments/14/best_checkpoint.pt --n_vl 5000
    APEROL_BASE=/scratch/yqw/aperol python3 eval_checkpoint.py experiments/14/best_checkpoint.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

BASE = os.environ.get("APEROL_BASE", "/scratch/yqw/aperol")
sys.path.insert(0, BASE)

from aperol.data.md17 import load_md17, collate_md17


def _import_model_class(exp_dir: Path) -> None:
    """Load Model/Layer class definitions from the experiment's run.py into __main__."""
    import importlib.util
    run_py = exp_dir / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"run.py not found in {exp_dir}")
    spec = importlib.util.spec_from_file_location("_exp_run", run_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["__main__"] = mod
    spec.loader.exec_module(mod)


def evaluate(ckpt_path: Path, data: str, n_tr: int, n_vl: int,
             batch_size: int) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ckpt["model"].to(device)
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

    return {
        "val_force_mse": val_force_sum / val_n,
        "val_energy_mse": val_energy_sum / val_n,
        "n_val": val_n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="Path to checkpoint.pt or best_checkpoint.pt")
    parser.add_argument("--data", default="malonaldehyde")
    parser.add_argument("--n_tr", type=int, default=1000)
    parser.add_argument("--n_vl", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--base-dir", default=None, help=f"Override aperol base (default: {BASE})")
    args = parser.parse_args()

    global BASE
    if args.base_dir:
        BASE = args.base_dir
        sys.path.insert(0, BASE)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(BASE) / "scripts/md17" / ckpt_path

    print(f"Evaluating: {ckpt_path}", flush=True)

    # Load class definitions from the experiment's run.py
    exp_dir = ckpt_path.parent
    _import_model_class(exp_dir)

    results = evaluate(ckpt_path, args.data, args.n_tr, args.n_vl, args.batch_size)

    print(f"\nResults (n_val={results['n_val']}):")
    print(f"  val_force_mse  = {results['val_force_mse']:.4f}")
    print(f"  val_energy_mse = {results['val_energy_mse']:.4f}")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
