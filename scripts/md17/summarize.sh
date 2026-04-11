#!/bin/bash
# Show a summary table of all experiment results.
# Usage: bash scripts/md17/summarize.sh [experiment_dir]
# Default: reads from /scratch/yqw/aperol/scripts/md17/experiments/ on trillium via SSH.
# Or: bash scripts/md17/summarize.sh ./experiments  (local directory)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/summarize.py"

EXP_DIR="${1:-}"

if [[ -z "${EXP_DIR}" ]]; then
  # Run on remote cluster via master socket
  SOCK="${HOME}/.config/submitter/sockets/trillium.sock"
  HOST="yqw@trillium-gpu.scinet.utoronto.ca"
  REMOTE_BASE="/scratch/yqw/aperol"
  REMOTE_PY="${REMOTE_BASE}/scripts/md17/summarize.py"
  REMOTE_EXP="${REMOTE_BASE}/scripts/md17/experiments"

  if [[ ! -S "${SOCK}" ]]; then
    echo "Not connected to trillium. Run 'submitter connect' first." >&2
    exit 1
  fi

  # Detect the currently-running aperol experiment from squeue
  RUNNING_EXP=""
  running_raw=$(ssh -o ControlMaster=no -o "ControlPath=${SOCK}" -o BatchMode=yes "${HOST}" \
    "squeue -u yqw --noheader -o '%j' 2>/dev/null | grep '^aperol_exp' | head -1" 2>/dev/null || true)
  if [[ "${running_raw}" =~ ^aperol_exp([0-9]+)$ ]]; then
    RUNNING_EXP="${BASH_REMATCH[1]}"
  fi

  # Upload the Python script and run it remotely, passing the running experiment number
  scp -o ControlMaster=no -o "ControlPath=${SOCK}" -o BatchMode=yes \
    "${PY}" "${HOST}:${REMOTE_PY}" 2>/dev/null

  ssh -o ControlMaster=no -o "ControlPath=${SOCK}" -o BatchMode=yes "${HOST}" \
    "python3 '${REMOTE_PY}' '${REMOTE_EXP}' '${RUNNING_EXP}'"

else
  python3 "${PY}" "${EXP_DIR}"
fi
