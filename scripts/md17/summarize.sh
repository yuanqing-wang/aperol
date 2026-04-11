#!/bin/bash
# Show best and latest val_force for each experiment directory.
# Usage: bash scripts/md17/summarize.sh [experiment_dir]
# Default: reads from /scratch/yqw/aperol/scripts/md17/experiments/ on trillium via SSH.
# Or: bash scripts/md17/summarize.sh ./experiments  (local directory)

set -euo pipefail

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
    "$(cat <<'EOF'
python3 -c "
import os, json, glob

exp_dir = '/scratch/yqw/aperol/scripts/md17/experiments'
exps = sorted([d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))], key=int)

print(f'{'Exp':>5}  {'Best val_force':>14}  {'@epoch':>6}  {'Final val_force':>15}  {'Epochs':>6}')
print('-' * 58)

for e in exps:
    m = os.path.join(exp_dir, e, 'metrics.jsonl')
    if not os.path.exists(m): continue
    lines = [json.loads(l) for l in open(m)]
    if not lines: continue
    best = min(lines, key=lambda x: x['val_force_error'])
    last = lines[-1]
    print(f'{e:>5}  {best[\"val_force_error\"]:>14.4f}  {best[\"epoch\"]:>6d}  {last[\"val_force_error\"]:>15.4f}  {len(lines):>6d}')
"
EOF
)"

else
  # Local mode
  python3 -c "
import os, json

exps = sorted([d for d in os.listdir('${EXP_DIR}') if os.path.isdir(os.path.join('${EXP_DIR}', d))], key=int)

print(f\"{'Exp':>5}  {'Best val_force':>14}  {'@epoch':>6}  {'Final val_force':>15}  {'Epochs':>6}\")
print('-' * 58)

for e in exps:
    m = os.path.join('${EXP_DIR}', e, 'metrics.jsonl')
    if not os.path.exists(m): continue
    lines = [json.loads(l) for l in open(m)]
    if not lines: continue
    best = min(lines, key=lambda x: x['val_force_error'])
    last = lines[-1]
    print(f\"{e:>5}  {best['val_force_error']:>14.4f}  {best['epoch']:>6d}  {last['val_force_error']:>15.4f}  {len(lines):>6d}\")
"
fi
