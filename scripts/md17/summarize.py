#!/usr/bin/env python3
"""Print a summary table of experiment results from metrics.jsonl files.

Usage:
    python3 summarize.py <experiments_dir>

Columns:
    Exp       — experiment number
    BestVal   — best val_force_error across all epochs
    @ep       — epoch at which best val_force occurred
    TrainF    — train_force_error at the best-val epoch
    Ratio     — val/train ratio at the best-val epoch (proxy for overfitting)
    FinalVal  — val_force_error at the last recorded epoch
    Ep        — total epochs recorded
"""

import json
import os
import sys


def main(exp_dir: str) -> None:
    entries = sorted(
        [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))],
        key=lambda x: int(x),
    )

    hdr = f"{'Exp':>4}  {'BestVal':>8}  {'@ep':>4}  {'TrainF':>7}  {'Ratio':>6}  {'FinalVal':>9}  {'Ep':>3}"
    print(hdr)
    print("-" * len(hdr))

    for e in entries:
        m = os.path.join(exp_dir, e, "metrics.jsonl")
        if not os.path.exists(m):
            continue
        lines = [json.loads(line) for line in open(m)]
        if not lines:
            continue
        best = min(lines, key=lambda x: x["val_force_error"])
        last = lines[-1]
        tf = best["train_force_error"]
        ratio = best["val_force_error"] / tf if tf > 0 else float("nan")
        print(
            f"{e:>4}  {best['val_force_error']:>8.4f}  {best['epoch']:>4d}  "
            f"{tf:>7.4f}  {ratio:>6.2f}  "
            f"{last['val_force_error']:>9.4f}  {len(lines):>3d}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <experiments_dir>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
