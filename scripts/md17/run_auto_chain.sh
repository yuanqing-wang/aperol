#!/bin/bash
# Continuously run optimizer resets until val_force_error < target.
# Creates each new experiment using new_reset_exp.sh, then submits and polls.
# Stops early if the target is reached.
#
# Usage: bash scripts/md17/run_auto_chain.sh <start_exp> [--target <val>] [--max <n>]
#   start_exp  — start the chain from this experiment (must have a checkpoint)
#   --target   — stop when val_force < this value (default: 1.0)
#   --max      — maximum number of new experiments to create (default: 30)
#
# Example: bash scripts/md17/run_auto_chain.sh 9 --target 1.0 --max 20

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMITTER="${HOME}/Documents/GitHub/submitter/submitter"
NEW_RESET="${SCRIPT_DIR}/new_reset_exp.sh"
SUMMARIZE="${SCRIPT_DIR}/summarize.sh"
REMOTE_BASE="/scratch/yqw/aperol/scripts/md17/experiments"
CLUSTER="trillium"
POLL_INTERVAL=60

TARGET=1.0
MAX_NEW=30
START_EXP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --max)    MAX_NEW="$2"; shift 2 ;;
    *)        START_EXP="$1"; shift ;;
  esac
done

[[ -z "${START_EXP}" ]] && { echo "Usage: $0 <start_exp> [--target <val>] [--max <n>]" >&2; exit 1; }

# Pre-flight: verify the start experiment has a checkpoint (must be finished)
SOCK="${HOME}/.config/submitter/sockets/trillium.sock"
HOST="yqw@trillium-gpu.scinet.utoronto.ca"
START_CKPT="${REMOTE_BASE}/${START_EXP}/best_checkpoint.pt"
START_CKPT_ALT="${REMOTE_BASE}/${START_EXP}/checkpoint.pt"
if ! ssh -o ControlMaster=no -o "ControlPath=${SOCK}" -o BatchMode=yes "${HOST}" \
    "test -f '${START_CKPT}' || test -f '${START_CKPT_ALT}'" 2>/dev/null; then
  echo "ERROR: exp${START_EXP} has no checkpoint. Make sure it has completed before running auto-chain." >&2
  exit 1
fi

echo "Auto-chain: start=exp${START_EXP}, target val_force < ${TARGET}, max ${MAX_NEW} new experiments"
echo ""

PREV="${START_EXP}"
for ((i = 1; i <= MAX_NEW; i++)); do
  NEW=$(( PREV + 1 ))
  echo "=== Creating exp${NEW} (reset from exp${PREV}) ==="

  # Create the new experiment
  bash "${NEW_RESET}" "${PREV}" "${NEW}"

  echo ""
  echo "=== Submitting exp${NEW} ==="

  # Submit and get job ID
  submit_out=$("${SUBMITTER}" submit-remote "${CLUSTER}" "${REMOTE_BASE}/${NEW}/job.sh" 2>&1)
  echo "${submit_out}"
  job_id=$(echo "${submit_out}" | grep -oE '[0-9]+$' | tail -1)

  if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse job ID" >&2
    exit 1
  fi

  echo "Job ${job_id} submitted for exp${NEW}. Polling every ${POLL_INTERVAL}s ..."

  if ! "${SUBMITTER}" poll "${CLUSTER}" "${job_id}" -i "${POLL_INTERVAL}"; then
    echo "ERROR: exp${NEW} (job ${job_id}) did not complete successfully." >&2
    exit 1
  fi

  echo "exp${NEW} complete."

  # Show current summary
  bash "${SUMMARIZE}" 2>/dev/null || true

  # Check if we've reached the target val_force
  SOCK="${HOME}/.config/submitter/sockets/trillium.sock"
  HOST="yqw@trillium-gpu.scinet.utoronto.ca"
  best_val=$(ssh -o ControlMaster=no -o "ControlPath=${SOCK}" -o BatchMode=yes "${HOST}" \
    "python3 -c \"
import json, os
m = '${REMOTE_BASE}/${NEW}/metrics.jsonl'
lines = [json.loads(l) for l in open(m)]
print(min(l['val_force_error'] for l in lines))
\"" 2>/dev/null || echo "999")

  echo ""
  echo "Best val_force for exp${NEW}: ${best_val}"

  # Compare with target (use awk for float comparison)
  if awk "BEGIN { exit !(${best_val} < ${TARGET}) }"; then
    echo ""
    echo "TARGET REACHED: val_force=${best_val} < ${TARGET}"
    echo "Chain complete after ${i} new experiments."
    bash "${SUMMARIZE}" 2>/dev/null || true
    exit 0
  fi

  PREV="${NEW}"
done

echo ""
echo "Reached max experiments (${MAX_NEW}). Best val_force was ${best_val}."
bash "${SUMMARIZE}" 2>/dev/null || true
