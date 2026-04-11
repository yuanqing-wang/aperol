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

echo "Creating exp${NEW} (optimizer reset from exp${PREV})..."
echo "  init_from: ${INIT_FROM}"

# Create directory and copy run.py
ssh -o ControlMaster=no -o "ControlPath=${SOCK}" -o BatchMode=yes "${HOST}" "
  mkdir -p '${REMOTE_NEW}'
  cp '${REMOTE_BASE}/${PREV}/run.py' '${REMOTE_NEW}/run.py'
  echo '# Docstring auto-updated by new_reset_exp.sh' > /dev/null
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
