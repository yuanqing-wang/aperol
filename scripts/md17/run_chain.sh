#!/bin/bash
# Sequentially submit a chain of SLURM experiments on Trillium.
# Waits for each job to complete before submitting the next.
# Compatible with the debug partition (no pending jobs allowed).
#
# Usage: bash scripts/md17/run_chain.sh <exp_n> [<exp_m> ...]
# Example: bash scripts/md17/run_chain.sh 6 7 8
#
# Requirements:
#   - submitter connect must have been run (master socket open)
#   - Each experiments/{n}/job.sh must exist on the cluster

set -euo pipefail

SUBMITTER="${HOME}/Documents/GitHub/submitter/submitter"
REMOTE_BASE="/scratch/yqw/aperol/scripts/md17/experiments"
CLUSTER="trillium"
POLL_INTERVAL=30

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <exp_n> [<exp_m> ...]"
  echo "Example: $0 6 7 8"
  exit 1
fi

for n in "$@"; do
  job_sh="${REMOTE_BASE}/${n}/job.sh"
  echo ""
  echo "=== Submitting exp ${n} ==="

  # Submit the job; parse the job ID from sbatch output
  submit_out=$("${SUBMITTER}" submit-remote "${CLUSTER}" "${job_sh}" 2>&1)
  echo "${submit_out}"
  job_id=$(echo "${submit_out}" | grep -oE '[0-9]+$' | tail -1)

  if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse job ID from submit output" >&2
    exit 1
  fi

  echo "Job ${job_id} submitted. Polling every ${POLL_INTERVAL}s ..."
  if ! "${SUBMITTER}" poll "${CLUSTER}" "${job_id}" -i "${POLL_INTERVAL}"; then
    echo "WARNING: job ${job_id} did not complete successfully (state not COMPLETED)." >&2
    echo "Check logs before continuing." >&2
    exit 1
  fi

  echo "Exp ${n} complete (job ${job_id})."

  # Print a quick summary after each experiment
  bash "$(dirname "$0")/summarize.sh" 2>/dev/null || true
done

echo ""
echo "Chain complete: experiments $* finished."
