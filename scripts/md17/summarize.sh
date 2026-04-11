#!/bin/bash
# Show a summary table of all experiment results.
# Usage: bash scripts/md17/summarize.sh [experiment_dir]
# Default: reads from /scratch/yqw/aperol/scripts/md17/experiments/ on trillium via SSH.
# Or: bash scripts/md17/summarize.sh ./experiments  (local directory)

set -euo pipefail

PYTHON_SCRIPT='
import os, json, glob, sys

exp_dir = sys.argv[1]
exps = sorted(
    [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))],
    key=lambda x: int(x)
)

hdr = f"{'Exp':>4}  {'BestVal':>8}  {'@ep':>4}  {'TrainF':>7}  {'Ratio':>6}  {'FinalVal':>9}  {'Ep':>3}"
print(hdr)
print("-" * len(hdr))

for e in exps:
    m = os.path.join(exp_dir, e, "metrics.jsonl")
    if not os.path.exists(m):
        continue
    lines = [json.loads(l) for l in open(m)]
    if not lines:
        continue
    best = min(lines, key=lambda x: x["val_force_error"])
    last = lines[-1]
    ratio = best["val_force_error"] / best["train_force_error"] if best["train_force_error"] > 0 else float("nan")
    print(f"{e:>4}  {best[\"val_force_error\"]:>8.4f}  {best[\"epoch\"]:>4d}  "
          f"{best[\"train_force_error\"]:>7.4f}  {ratio:>6.2f}  "
          f"{last[\"val_force_error\"]:>9.4f}  {len(lines):>3d}")
'

EXP_DIR="${1:-}"

if [[ -z "${EXP_DIR}" ]]; then
  # Run on remote cluster via master socket
  SOCK="${HOME}/.config/submitter/sockets/trillium.sock"
  HOST="yqw@trillium-gpu.scinet.utoronto.ca"
  REMOTE_DIR="/scratch/yqw/aperol/scripts/md17/experiments"

  if [[ ! -S "${SOCK}" ]]; then
    echo "Not connected to trillium. Run 'submitter connect' first." >&2
    exit 1
  fi

  ssh -o ControlMaster=no -o "ControlPath=${SOCK}" -o BatchMode=yes "${HOST}" \
    "python3 -c '$( echo "${PYTHON_SCRIPT}" | sed "s/'/'\'''/g" )' '${REMOTE_DIR}'"

else
  python3 -c "${PYTHON_SCRIPT}" "${EXP_DIR}"
fi
