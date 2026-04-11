#!/bin/bash
# Create the next optimizer-reset experiment from a previous experiment's checkpoint.
# Copies run.py (preserving class definitions) and writes a ready-to-submit job.sh.
#
# Usage: bash scripts/md17/new_reset_exp.sh <prev_exp_n> [<new_exp_n>]
#   prev_exp_n  — experiment to reset from (uses best_checkpoint.pt if present)
#   new_exp_n   — new experiment number (default: prev_exp_n + 1)
#
# Example: bash scripts/md17/new_reset_exp.sh 6        # creates exp 7
#          bash scripts/md17/new_reset_exp.sh 6 9      # creates exp 9 from exp 6

set -euo pipefail

SOCK="${HOME}/.config/submitter/sockets/trillium.sock"
HOST="yqw@trillium-gpu.scinet.utoronto.ca"
REMOTE_BASE="/scratch/yqw/aperol/scripts/md17/experiments"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <prev_exp_n> [<new_exp_n>]" >&2
  exit 1
fi

PREV="$1"
NEW="${2:-$((PREV + 1))}"

# Prefer best_checkpoint.pt over checkpoint.pt
INIT_FROM="${REMOTE_BASE}/${PREV}/best_checkpoint.pt"
# Check if best_checkpoint exists on cluster
if ! ssh -o ControlMaster=no -o "ControlPath=${SOCK}" -o BatchMode=yes "${HOST}" \
    "test -f '${INIT_FROM}'" 2>/dev/null; then
  INIT_FROM="${REMOTE_BASE}/${PREV}/checkpoint.pt"
  echo "Note: best_checkpoint.pt not found for exp${PREV}, using checkpoint.pt"
fi

REMOTE_NEW="${REMOTE_BASE}/${NEW}"

# Guard: warn if destination already has a checkpoint (would overwrite a running/done exp)
if ssh -o ControlMaster=no -o "ControlPath=${SOCK}" -o BatchMode=yes "${HOST}" \
    "test -f '${REMOTE_NEW}/checkpoint.pt'" 2>/dev/null; then
  echo "WARNING: ${REMOTE_NEW}/checkpoint.pt already exists." >&2
  echo "  This experiment may already be running or completed." >&2
  if [[ -t 0 ]]; then
    # Interactive terminal: ask for confirmation
    read -r -p "  Overwrite? [y/N] " confirm
    [[ "${confirm}" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
  else
    # Non-interactive (e.g. called from run_chain.sh): abort to be safe
    echo "  Non-interactive mode — aborting to avoid overwrite. Remove checkpoint first." >&2
    exit 1
  fi
fi

echo "Creating exp${NEW} (optimizer reset from exp${PREV})..."
echo "  init_from: ${INIT_FROM}"

# Create directory, copy run.py, update first docstring line
ssh -o ControlMaster=no -o "ControlPath=${SOCK}" -o BatchMode=yes "${HOST}" "
  mkdir -p '${REMOTE_NEW}'
  cp '${REMOTE_BASE}/${PREV}/run.py' '${REMOTE_NEW}/run.py'
  sed -i '1s|^\"\"\".*$|\"\"\"Exp ${NEW} — Optimizer reset from exp${PREV} best_checkpoint.|' '${REMOTE_NEW}/run.py'
"

# Write job.sh
ssh -o ControlMaster=no -o "ControlPath=${SOCK}" -o BatchMode=yes "${HOST}" \
  "cat > '${REMOTE_NEW}/job.sh'" << EOF
#!/bin/bash
#SBATCH -J aperol_exp${NEW}
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH -n 1
#SBATCH -o ${REMOTE_NEW}/job_%j.log
#SBATCH -e ${REMOTE_NEW}/job_%j.err

set -euo pipefail
export PYTHONPATH=/scratch/yqw/aperol
export WANDB_MODE=offline
export WANDB_DIR=/scratch/yqw/aperol/scripts/md17/wandb
conda run -n aperol python -u ${REMOTE_NEW}/run.py \\
  --n_epoch 200 \\
  --learning_rate 1e-5 \\
  --scheduler_step 20 \\
  --init_from ${INIT_FROM} \\
  --checkpoint ${REMOTE_NEW}/checkpoint.pt
echo APEROL_JOB_DONE
EOF

echo "Created exp${NEW} with 200-epoch job.sh (2h time limit)."
echo "Submit with:"
echo "  ~/Documents/GitHub/submitter/submitter submit-remote trillium ${REMOTE_NEW}/job.sh"
