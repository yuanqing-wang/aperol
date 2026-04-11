#!/bin/bash
# Remove experiment directories from the cluster to free disk space.
# Keeps the best_checkpoint.pt and metrics.jsonl for the record, removes
# the large checkpoint.pt (model weights).
#
# Usage: bash scripts/md17/clean_exps.sh [options] <exp_n> [<exp_m> ...]
#   --rm-all    Remove the entire experiment directory (checkpoints + logs)
#   --rm-final  Remove only checkpoint.pt (not best_checkpoint.pt)
#   --dry-run   Show what would be removed without removing
#
# Examples:
#   bash scripts/md17/clean_exps.sh 1 2 4 5 6 7 8     # remove final checkpoints
#   bash scripts/md17/clean_exps.sh --rm-all 1 2 3    # remove entire dirs
#   bash scripts/md17/clean_exps.sh --dry-run 1 2     # preview

set -euo pipefail

SUBMITTER="${HOME}/Documents/GitHub/submitter/submitter"
CLUSTER="trillium"
REMOTE_BASE="/scratch/yqw/aperol/scripts/md17/experiments"
MODE="rm-final"  # rm-final | rm-all
DRY_RUN=0
EXPS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rm-all)    MODE="rm-all";  shift ;;
    --rm-final)  MODE="rm-final"; shift ;;
    --dry-run|-n) DRY_RUN=1;    shift ;;
    -*)          echo "Unknown option: $1" >&2; exit 1 ;;
    *)           EXPS+=("$1");  shift ;;
  esac
done

[[ ${#EXPS[@]} -eq 0 ]] && { echo "Usage: $0 [--rm-all|--rm-final] [--dry-run] <exp_n> ..." >&2; exit 1; }

echo "Mode: ${MODE}, dry-run: ${DRY_RUN}"
echo ""

for exp in "${EXPS[@]}"; do
  dir="${REMOTE_BASE}/${exp}"
  echo "  exp${exp}: ${dir}"

  # Check what exists
  contents=$("${SUBMITTER}" run-cmd "${CLUSTER}" "ls '${dir}/' 2>/dev/null" 2>/dev/null || echo "(dir not found)")
  echo "    Contents: $(echo "${contents}" | tr '\n' ' ')"

  if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ "${MODE}" == "rm-all" ]]; then
      echo "    WOULD remove entire directory"
    else
      echo "    WOULD remove checkpoint.pt (keep best_checkpoint.pt + metrics.jsonl)"
    fi
    continue
  fi

  if [[ "${MODE}" == "rm-all" ]]; then
    "${SUBMITTER}" run-cmd "${CLUSTER}" "rm -rf '${dir}'" 2>/dev/null
    echo "    Removed entire directory"
  else
    # rm-final: remove only checkpoint.pt (large model weights)
    "${SUBMITTER}" run-cmd "${CLUSTER}" "rm -f '${dir}/checkpoint.pt'" 2>/dev/null
    echo "    Removed checkpoint.pt (best_checkpoint.pt and metrics.jsonl preserved)"
  fi
done

echo ""
echo "Done. Disk usage:"
"${SUBMITTER}" run-cmd "${CLUSTER}" \
  "du -sh '${REMOTE_BASE}' 2>/dev/null || echo '(unavailable)'" 2>/dev/null
