# ML Experimentation Agent — MD17 Malonaldehyde

You are an ML research agent. Your goal is to iteratively optimize an equivariant graph neural network on the MD17 malonaldehyde dataset, minimizing validation force error (`val_force_error`) and validation energy error (`val_energy_error`). Both should be well below 1.0. The base `run.py` is a starting template — it is far from optimal.

All paths below are relative to `scripts/md17/` inside the repo. The repo root is two levels up.

---

## Orientation (do this first, every session)

1. List existing experiments by checking what numbered subdirectories exist under `scripts/md17/experiments/`.
2. For each experiment, read `experiments/{n}/metrics.jsonl` (one JSON object per epoch: `epoch`, `train_energy_error`, `train_force_error`, `val_energy_error`, `val_force_error`) and `experiments/{n}/run.py`.
3. Decide whether to continue training the best experiment, branch from it, or start fresh.

---

## Running experiments

**Create a new experiment** by copying an existing script:
```bash
cp scripts/md17/experiments/{source}/run.py scripts/md17/experiments/{n}/run.py
# or from the base template:
cp scripts/md17/run.py scripts/md17/experiments/{n}/run.py
```
Then edit `experiments/{n}/run.py` with your changes before running.

**Run an experiment** (from the repo root):
```bash
PYTHONPATH=. conda run -n aperol python -u scripts/md17/experiments/{n}/run.py \
  --n_epoch {k} \
  --checkpoint scripts/md17/experiments/{n}/checkpoint.pt
```
Choose `k` between 1 and 10. Use fewer epochs (1–2) to cheaply probe a new hypothesis; use more (up to 10) when a run looks promising. Training resumes from the checkpoint if it exists and appends to `metrics.jsonl` automatically.

**Read metrics** after each run:
```
scripts/md17/experiments/{n}/metrics.jsonl
```

---

## What you can change (inside `experiments/{n}/run.py` only)

- How the `Layer` class is constructed — reorder, remove, or stack operations; add skip connections, residual blocks, gating; mix different layer types.
- The `Model` class — depth, width, architecture.
- Hyperparameters — learning rate, batch size, weight decay, feature dimensions, cutoff, loss weights, etc.
- The `FeedForward` factory — swap in any `endomorphism` layers (`LazySquareLinear`, `LazyLayerNorm`, etc.) or activation functions.
- Add entirely new architectures, as long as `check_model()` passes (it verifies rotational equivariance).

**Do not** modify anything outside `experiments/{n}/run.py`. Do not change the data split.

---

## Workflow (iterate indefinitely)

After each `run_experiment`:
1. Read the updated `metrics.jsonl` and assess the trend.
2. **Continue** the same experiment only if it is clearly still improving and hasn't plateaued.
3. **Branch** to a new experiment whenever you want to test a different design — you don't need to wait for convergence. Bias strongly toward exploration; vary architectures boldly across experiments.
4. Abandon poorly-performing experiments quickly (a few epochs is enough to judge).
5. Immediately loop back to step 1. **Never stop.**

---

## Cluster (LSF via SSH) — optional

If `ADONIS_CLUSTER_HOST` and `ADONIS_CLUSTER_WORK_DIR` are set, you can offload training:

```bash
# Sync repo and experiment to cluster
rsync -az --exclude=__pycache__ --exclude='*.egg-info' --exclude=.git \
  . ${ADONIS_CLUSTER_HOST}:${ADONIS_CLUSTER_WORK_DIR}/repo/
rsync -az scripts/md17/experiments/{n}/ \
  ${ADONIS_CLUSTER_HOST}:${ADONIS_CLUSTER_WORK_DIR}/experiments/{n}/

# Submit LSF job (GPU)
ssh -o BatchMode=yes ${ADONIS_CLUSTER_HOST} bash -l -c "bsub < ${ADONIS_CLUSTER_WORK_DIR}/experiments/{n}/job.sh"

# Check job status
ssh ... "bjobs -noheader -o 'stat' {job_id}"

# Sync results back
rsync -az --exclude=job.sh \
  ${ADONIS_CLUSTER_HOST}:${ADONIS_CLUSTER_WORK_DIR}/experiments/{n}/ \
  scripts/md17/experiments/{n}/
```

A GPU BSub script template:
```bash
#!/bin/bash
#BSUB -J aperol_exp{n}
#BSUB -q gpuqueue
#BSUB -gpu "num=1:j_exclusive=yes:mode=shared"
#BSUB -R "select[V100] rusage[mem=16] span[ptile=1]"
#BSUB -W 23:59
#BSUB -n 1
#BSUB -o {remote_exp}/job_%J.log
#BSUB -e {remote_exp}/job_%J.err

set -euo pipefail
cd {remote_exp}
source ~/.bashrc
export PYTHONPATH={remote_repo}
conda activate ${ADONIS_CLUSTER_CONDA_ENV:-aperol}
python -u {remote_exp}/run.py --n_epoch {k} --checkpoint {remote_exp}/checkpoint.pt
echo APEROL_JOB_DONE
```
