#!/bin/bash
# Create the next optimizer-reset experiment from a previous experiment's checkpoint.
# Copies run.py (preserving class definitions) and writes a ready-to-submit job.sh.
#
# Usage: bash scripts/md17/new_reset_exp.sh <prev_exp_n> [<new_exp_n>] [options]
#   prev_exp_n       — experiment to reset from (uses best_checkpoint.pt if present)
#   new_exp_n        — new experiment number (default: prev_exp_n + 1)
#   --use-run N      — copy run.py from exp N instead of prev_exp_n (avoids accidental
#                      architecture drift — CRITICAL if the experiment chain diverged)
#   --use-final-ckpt — use checkpoint.pt instead of best_checkpoint.pt as init_from.
#                      Use this when best_checkpoint.pt may be corrupt (e.g. it was
#                      written by a FAILED prior job that used a different architecture).
#
# Examples:
#   bash scripts/md17/new_reset_exp.sh 11 12           # exp12 from exp11/best
#   bash scripts/md17/new_reset_exp.sh 11 12 --use-final-ckpt  # use exp11/checkpoint.pt
#   bash scripts/md17/new_reset_exp.sh 9 12 --use-run 11  # exp12 from exp9 ckpt, exp11 arch

set -euo pipefail

SUBMITTER="${HOME}/Documents/GitHub/submitter/submitter"
CLUSTER="trillium"
SOCK="${HOME}/.config/submitter/sockets/trillium.sock"
HOST="yqw@trillium-gpu.scinet.utoronto.ca"
REMOTE_BASE="/scratch/yqw/aperol/scripts/md17/experiments"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <prev_exp_n> [<new_exp_n>] [--use-run <n>]" >&2
  exit 1
fi

PREV="$1"; shift
NEW=""
RUN_SOURCE=""
USE_FINAL_CKPT=0
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --use-run)        RUN_SOURCE="$2"; shift 2 ;;
    --use-final-ckpt) USE_FINAL_CKPT=1; shift ;;
    --extra-args)     EXTRA_ARGS="$2"; shift 2 ;;
    --weight-noise)   EXTRA_ARGS="${EXTRA_ARGS} --weight_noise_std $2"; shift 2 ;;
    --clip-grad)      EXTRA_ARGS="${EXTRA_ARGS} --clip_grad_norm $2"; shift 2 ;;
    --adamw)          EXTRA_ARGS="${EXTRA_ARGS} --optimizer adamw"; shift ;;
    --plateau)        EXTRA_ARGS="${EXTRA_ARGS} --scheduler plateau"; shift ;;
    *) NEW="$1"; shift ;;
  esac
done
[[ -z "${NEW}" ]] && NEW=$(( PREV + 1 ))
[[ -z "${RUN_SOURCE}" ]] && RUN_SOURCE="${PREV}"

# Prefer best_checkpoint.pt over checkpoint.pt, unless --use-final-ckpt is set
# (use --use-final-ckpt if best_checkpoint.pt may be corrupt from a failed prior job)
if [[ "${USE_FINAL_CKPT}" == "1" ]]; then
  INIT_FROM="${REMOTE_BASE}/${PREV}/checkpoint.pt"
  echo "Note: using final checkpoint.pt (not best_checkpoint.pt)"
else
  INIT_FROM="${REMOTE_BASE}/${PREV}/best_checkpoint.pt"
  # Check if best_checkpoint exists on cluster (via submitter)
  if ! "${SUBMITTER}" run-cmd "${CLUSTER}" "test -f '${INIT_FROM}'" 2>/dev/null; then
    INIT_FROM="${REMOTE_BASE}/${PREV}/checkpoint.pt"
    echo "Note: best_checkpoint.pt not found for exp${PREV}, using checkpoint.pt"
  fi
fi

REMOTE_NEW="${REMOTE_BASE}/${NEW}"

# Guard: warn if destination already has a checkpoint (would overwrite a running/done exp)
if "${SUBMITTER}" run-cmd "${CLUSTER}" "test -f '${REMOTE_NEW}/checkpoint.pt'" 2>/dev/null; then
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
# NOTE: run.py is copied from RUN_SOURCE (default=PREV), not always PREV.
#       Use --use-run N to pick a specific architecture when chains diverge.
echo "  run.py source: exp${RUN_SOURCE}"
"${SUBMITTER}" run-cmd "${CLUSTER}" "
  mkdir -p '${REMOTE_NEW}'
  cp '${REMOTE_BASE}/${RUN_SOURCE}/run.py' '${REMOTE_NEW}/run.py'
  sed -i '1s|^\"\"\".*$|\"\"\"Exp ${NEW} — Optimizer reset from exp${PREV} best_checkpoint.|' '${REMOTE_NEW}/run.py'
"

# Write job.sh (must use submitter run-cmd with stdin for cat heredoc)
"${SUBMITTER}" run-cmd "${CLUSTER}" "cat > '${REMOTE_NEW}/job.sh'" << EOF
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

# Verify init_from checkpoint exists before starting Python
if [ ! -f "${INIT_FROM}" ]; then
  echo "ERROR: init_from checkpoint not found: ${INIT_FROM}" >&2
  exit 1
fi

conda run -n aperol python -u ${REMOTE_NEW}/run.py \\
  --n_epoch 200 \\
  --learning_rate 1e-5 \\
  --scheduler_step 20 \\
  --init_from ${INIT_FROM} \\
  --checkpoint ${REMOTE_NEW}/checkpoint.pt${EXTRA_ARGS:+ ${EXTRA_ARGS}}
echo APEROL_JOB_DONE
EOF

echo "Created exp${NEW} with 200-epoch job.sh (2h time limit)."
echo "Submit with:"
echo "  ~/Documents/GitHub/submitter/submitter submit-remote trillium ${REMOTE_NEW}/job.sh"
