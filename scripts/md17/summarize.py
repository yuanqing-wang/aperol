#!/usr/bin/env python3
"""Print a summary table of experiment results from metrics.jsonl files.

Usage:
    python3 summarize.py <experiments_dir> [running_exp_num]

Arguments:
    experiments_dir  — path to the experiments directory
    running_exp_num  — (optional) experiment number currently running; marked with *

Columns:
    Exp       — experiment number (* = currently running)
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


def _detect_actual_init(exp_dir: str, exp_name: str) -> str:
    """Look in job logs for 'Loaded model from' to detect actual checkpoint used."""
    import re
    import glob
    log_pattern = os.path.join(exp_dir, exp_name, "job_*.log")
    logs = sorted(glob.glob(log_pattern))
    for log in reversed(logs):  # most recent log first
        try:
            txt = open(log, errors="ignore").read(5000)
            m = re.search(r"Loaded model from (.+), fresh optimizer", txt)
            if m:
                return m.group(1)  # full path of init_from
            m = re.search(r"Resumed from (.+) at epoch", txt)
            if m:
                return m.group(1) + " (resume)"
        except Exception:
            pass
    return ""


def _detect_chain(exp_dir: str, exp_name: str) -> str:
    """Return a short chain label by scanning the run.py imports and class bodies.

    For cases where actual runtime behavior differs from code (e.g. checkpoint mixup),
    the code-based label may be misleading — check job logs separately.
    """
    import re
    run_py = os.path.join(exp_dir, exp_name, "run.py")
    if not os.path.exists(run_py):
        return ""
    txt = open(run_py, errors="ignore").read()
    # Only check actual code lines (not docstrings) by looking for import or attribute assignment
    has_sender = bool(re.search(r"^\s*(from|import).*NodeToEdgeSenderBroadcast", txt, re.MULTILINE) or
                      re.search(r"NodeToEdgeSenderBroadcast\s*\(", txt))
    has_edge_pos = bool(re.search(r"EdgeToPositionAggregation\s*\(", txt))
    if has_sender and has_edge_pos:
        return "B+"  # bold: sender+receiver + edge→pos/vel
    if has_sender:
        return "B"   # chain B: sender+receiver (note: verify actual init via job log)
    return "A"       # chain A: receiver-only (base)


def main(exp_dir: str, running: str = "") -> None:
    entries = sorted(
        [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))],
        key=lambda x: int(x),
    )

    hdr = f"{'Exp':>5}  {'Ch':>3}  {'BestVal':>8}  {'@ep':>4}  {'TrainF':>7}  {'Ratio':>6}  {'FinalVal':>9}  {'Ep':>3}"
    print(hdr)
    print("-" * len(hdr))

    for e in entries:
        m = os.path.join(exp_dir, e, "metrics.jsonl")
        if not os.path.exists(m):
            # Show experiments with no metrics yet (e.g. staged)
            chain = _detect_chain(exp_dir, e)
            marker = "*" if e == running else " "
            print(f"{e:>4}{marker}  {chain:>3}  {'—':>8}  {'—':>4}  {'—':>7}  {'—':>6}  {'—':>9}  {'—':>3}  (staged)")
            continue
        lines = [json.loads(line) for line in open(m)]
        if not lines:
            continue
        best = min(lines, key=lambda x: x["val_force_error"])
        last = lines[-1]
        tf = best["train_force_error"]
        ratio = best["val_force_error"] / tf if tf > 0 else float("nan")
        chain = _detect_chain(exp_dir, e)
        marker = "*" if e == running else " "
        running_note = " ← running" if e == running else ""
        print(
            f"{e:>4}{marker}  {chain:>3}  {best['val_force_error']:>8.4f}  {best['epoch']:>4d}  "
            f"{tf:>7.4f}  {ratio:>6.2f}  "
            f"{last['val_force_error']:>9.4f}  {len(lines):>3d}{running_note}"
        )

    # Summary footer: best overall and progress to target
    import math
    best_vals_by_entry = {}
    for e in entries:
        m = os.path.join(exp_dir, e, "metrics.jsonl")
        if not os.path.exists(m):
            continue
        lines = [json.loads(l) for l in open(m)]
        if lines:
            best = min(lines, key=lambda x: x["val_force_error"])
            best_vals_by_entry[int(e)] = best["val_force_error"]

    if best_vals_by_entry:
        best_overall = min(best_vals_by_entry.values())
        target = 1.0

        # Estimate improvement rate from chain A experiments only (sorted by exp number)
        sorted_exps = sorted(best_vals_by_entry.keys())
        improvements = []
        prev_val = None
        for exp_n in sorted_exps:
            chain = _detect_chain(exp_dir, str(exp_n))
            v = best_vals_by_entry[exp_n]
            if chain == "A" and prev_val is not None and v < prev_val:
                improvements.append((prev_val - v) / prev_val)
            if chain == "A":
                prev_val = v  # only update prev from chain A
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0.091

        # Also compute recent improvement rate (last 3 chain A resets) for a realistic estimate
        recent = improvements[-3:] if len(improvements) >= 3 else improvements
        recent_improvement = sum(recent) / len(recent) if recent else avg_improvement

        # Per-chain bests
        chain_bests: dict[str, float] = {}
        for exp_n, v in best_vals_by_entry.items():
            ch = _detect_chain(exp_dir, str(exp_n))
            if ch not in chain_bests or v < chain_bests[ch]:
                chain_bests[ch] = v

        if best_overall > target:
            n_overall = math.ceil(math.log(target / best_overall) / math.log(1 - avg_improvement))
            n_recent = math.ceil(math.log(target / best_overall) / math.log(1 - recent_improvement))
            pct_avg = avg_improvement * 100
            pct_recent = recent_improvement * 100
            mins_per_reset = 67
            print(f"\nBest: {best_overall:.4f}  Target: <{target}")
            # Show per-chain bests when multiple chains exist
            if len(chain_bests) > 1:
                chain_summary = "  |  ".join(
                    f"Chain {ch}: {v:.4f}" for ch, v in sorted(chain_bests.items())
                )
                print(f"  Per chain: {chain_summary}")
            print(f"  Overall avg: ~{pct_avg:.1f}%/reset → ~{n_overall} resets "
                  f"(~{n_overall * mins_per_reset // 60}h)")
            print(f"  Recent (last 3): ~{pct_recent:.1f}%/reset → ~{n_recent} resets "
                  f"(~{n_recent * mins_per_reset // 60}h)  ← more realistic")
            # Actionable recommendation when recent rate is very slow
            if recent_improvement < 0.05 and "B+" in chain_bests:
                print(f"  ⚠ Chain A diminishing returns (<5%/reset). "
                      f"Consider switching to B+ chain (best: {chain_bests['B+']:.4f}).")

            # Show B+ improvement rate separately if available.
            # Exclude the currently-running experiment (its best is early/noisy)
            # unless it represents a complete reset (>= 50 epochs recorded).
            bplus_exps = sorted(
                [exp_n for exp_n in best_vals_by_entry
                 if _detect_chain(exp_dir, str(exp_n)) == "B+"],
            )
            # Remove the running experiment if it has few epochs (noisy best)
            if bplus_exps and running:
                try:
                    running_n = int(running)
                    if running_n == bplus_exps[-1]:
                        m = os.path.join(exp_dir, str(running_n), "metrics.jsonl")
                        n_epochs = sum(1 for _ in open(m)) if os.path.exists(m) else 0
                        if n_epochs < 50:
                            bplus_exps = bplus_exps[:-1]  # drop early-running exp
                except (ValueError, OSError):
                    pass
            if len(bplus_exps) >= 2:
                bplus_improvements = []
                for i in range(1, len(bplus_exps)):
                    prev_v = best_vals_by_entry[bplus_exps[i - 1]]
                    curr_v = best_vals_by_entry[bplus_exps[i]]
                    if curr_v < prev_v:
                        bplus_improvements.append((prev_v - curr_v) / prev_v)
                if bplus_improvements:
                    bplus_avg = sum(bplus_improvements) / len(bplus_improvements)
                    bplus_n = math.ceil(
                        math.log(target / chain_bests["B+"]) / math.log(1 - bplus_avg)
                    )
                    # Build projected trajectory
                    trajectory = [chain_bests["B+"]]
                    v = chain_bests["B+"]
                    while v > target and len(trajectory) <= bplus_n + 1:
                        v = v * (1 - bplus_avg)
                        trajectory.append(v)
                    traj_str = " → ".join(f"{v:.1f}" for v in trajectory[:8])
                    if len(trajectory) > 8:
                        traj_str += f" → ... → {trajectory[-1]:.2f}"
                    print(f"  B+ avg: ~{bplus_avg * 100:.1f}%/reset → ~{bplus_n} more resets "
                          f"(~{bplus_n * mins_per_reset // 60}h)")
                    print(f"  B+ projected: {traj_str} ({'✓' if trajectory[-1] < target else '✗'})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <experiments_dir> [running_exp_num]", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
