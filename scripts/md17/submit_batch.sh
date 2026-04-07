#!/bin/bash
# Batch submission script for experiments
# Usage: bash scripts/md17/submit_batch.sh [start_exp] [end_exp]
# Example: bash scripts/md17/submit_batch.sh 357 435

set -euo pipefail

HOST="wangy1@lilac.mskcc.org"
REMOTE_WORK="/data/chodera/wangyq/aperol"
CONDA_ENV="aperol"
N_EPOCH=200

START=${1:-357}
END=${2:-440}

echo "=== Syncing repo to cluster ==="
rsync -az --exclude=__pycache__ --exclude='*.egg-info' --exclude=.git --exclude='experiments/' \
  . ${HOST}:${REMOTE_WORK}/

echo ""
echo "=== Syncing and submitting experiments ${START}-${END} ==="

for n in $(seq ${START} ${END}); do
    EXP_DIR="scripts/md17/experiments/${n}"
    REMOTE_EXP="${REMOTE_WORK}/${EXP_DIR}"

    if [ ! -f "${EXP_DIR}/run.py" ]; then
        echo "SKIP exp${n}: no run.py"
        continue
    fi

    echo -n "exp${n}: syncing... "
    rsync -az "${EXP_DIR}/" "${HOST}:${REMOTE_EXP}/"

    echo -n "writing job.sh... "
    # Write job.sh remotely using printf to avoid heredoc quoting issues
    ssh -o BatchMode=yes -o ConnectTimeout=15 ${HOST} \
        "printf '%s\n' \
        '#!/bin/bash' \
        '#BSUB -J aperol_exp${n}' \
        '#BSUB -q gpuqueue' \
        '#BSUB -gpu \"num=1:j_exclusive=yes:mode=shared\"' \
        '#BSUB -R \"select[V100] rusage[mem=16] span[ptile=1]\"' \
        '#BSUB -W 23:59' \
        '#BSUB -n 1' \
        \"#BSUB -o ${REMOTE_EXP}/job_%J.log\" \
        \"#BSUB -e ${REMOTE_EXP}/job_%J.err\" \
        '' \
        'set -euo pipefail' \
        'source ~/.bashrc' \
        \"export PYTHONPATH=${REMOTE_WORK}\" \
        \"conda run -n ${CONDA_ENV} python -u ${REMOTE_EXP}/run.py \\\\\" \
        \"  --n_epoch ${N_EPOCH} \\\\\" \
        \"  --checkpoint ${REMOTE_EXP}/checkpoint.pt\" \
        'echo APEROL_JOB_DONE' \
        > ${REMOTE_EXP}/job.sh"

    echo -n "submitting... "
    JOB_OUTPUT=$(ssh -o BatchMode=yes -o ConnectTimeout=15 ${HOST} bash -l -c "\"bsub < ${REMOTE_EXP}/job.sh\"" 2>&1 || true)
    JOB_ID=$(echo "${JOB_OUTPUT}" | grep -oE 'Job <[0-9]+>' | grep -oE '[0-9]+' | head -1 || echo "unknown")
    echo "job_id=${JOB_ID}"

done

echo ""
echo "=== All experiments submitted ==="
echo "Check status with: ssh ${HOST} 'bjobs'"
