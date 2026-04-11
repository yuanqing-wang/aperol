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
    preds, targets = [], []
    for batch in val_loader:
        batch = batch.to(device)
        batch.position.requires_grad_(True)
        energy = model(batch)
        force = -torch.autograd.grad(energy.sum(), batch.position, create_graph=False)[0]
        B = batch.energy.shape[0]
        val_energy_sum += F.mse_loss(energy, batch.energy).item() * B
        val_force_sum += F.mse_loss(force, batch.force).item() * B
        val_n += B
        preds.append(energy.detach())
        targets.append(batch.energy.detach())

    # Calibrated energy error: fit a*pred + b = true via OLS, then compute MSE
    preds_all = torch.cat(preds)
    targets_all = torch.cat(targets)
    A = torch.stack([preds_all, torch.ones_like(preds_all)], dim=1)  # (N, 2)
    try:
        coeffs, _, _, _ = torch.linalg.lstsq(A, targets_all.unsqueeze(-1))
        a, b = coeffs[0, 0].item(), coeffs[1, 0].item()
        cal_preds = a * preds_all + b
        cal_energy_mse = F.mse_loss(cal_preds, targets_all).item()
    except Exception:
        a, b, cal_energy_mse = float("nan"), float("nan"), float("nan")

    return {
        "val_force_mse": val_force_sum / val_n,
        "val_energy_mse": val_energy_sum / val_n,
        "val_energy_cal_mse": cal_energy_mse,  # after linear calibration
        "cal_a": a, "cal_b": b,
        "n_val": val_n,
    }


def main():
    global BASE  # must be declared before any use of BASE in this function
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="Path to checkpoint.pt or best_checkpoint.pt")
    parser.add_argument("--data", default="malonaldehyde")
    parser.add_argument("--n_tr", type=int, default=1000)
    parser.add_argument("--n_vl", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--base-dir", default=None, help=f"Override aperol base (default: {BASE})")
    args = parser.parse_args()

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
    print(f"  val_force_mse      = {results['val_force_mse']:.4f}  (primary metric)")
    print(f"  val_energy_mse     = {results['val_energy_mse']:.4f}  (raw, includes constant offset)")
    print(f"  val_energy_cal_mse = {results['val_energy_cal_mse']:.4f}  (after linear calibration)")
    print(f"  calibration: E_true ≈ {results['cal_a']:.3f} * E_pred + {results['cal_b']:.3f}")

    # Cross-reference with metrics.jsonl to show overfitting ratio if available
    metrics_path = ckpt_path.parent / "metrics.jsonl"
    if metrics_path.exists():
        try:
            lines = [json.loads(l) for l in metrics_path.open()]
            if lines:
                # Find the entry closest in val_force to our evaluation
                best_m = min(lines, key=lambda x: abs(x["val_force_error"] - results["val_force_mse"]))
                tf = best_m.get("train_force_error")
                vf = best_m.get("val_force_error")
                ep = best_m.get("epoch")
                if tf and tf > 0:
                    print(f"\n  From metrics.jsonl (epoch {ep}): train_force={tf:.4f}, val_force={vf:.4f}, ratio={vf/tf:.2f}")
        except Exception:
            pass

    print(json.dumps(results))


if __name__ == "__main__":
    main()
