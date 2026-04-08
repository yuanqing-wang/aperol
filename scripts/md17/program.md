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

**Do not** change `n_vl` or `n_tr`. The validated defaults (`n_vl=1000`, `n_tr=1000`) must be preserved across all architecture experiments — changing them confounds comparisons.

**Be bold with architecture.** Past wins came from non-obvious structural changes (angle features, residuals, NequIP-style tensor products). When designing new experiments, lean toward high-variance ideas rather than small hyperparameter tweaks. Good candidates include: angle/dihedral features (3-body, 4-body), attention mechanisms, message-passing depth, multi-scale aggregation, tensor product layers, and combined best-of approaches. A bold experiment that fails fast is more informative than a cautious one that barely moves the needle.

---

## Insights log

Keep a running record of learned experiences in `scripts/md17/insight.md`. After every experiment (or whenever you draw a meaningful conclusion), append an entry in this format:

```markdown
## Exp {n} — <one-line description>
- **Result:** val_force_error=X.XX, val_energy_error=X.XX (epoch K)
- **What worked / didn't:** ...
- **Takeaway:** ...
```

Read `insight.md` at the start of each session (during Orientation) so prior findings inform new designs. Never overwrite the file — only append.

---

## Workflow (iterate indefinitely)

After each `run_experiment`:
1. **Preserve local copies.** Immediately after a run completes (local or cluster), ensure these two files exist locally:
   - `scripts/md17/experiments/{n}/run.py` — the exact script that was run
   - `scripts/md17/experiments/{n}/metrics.jsonl` — the full epoch log
   For cluster runs this means doing the rsync-back before anything else. For local runs the files are already in place.
2. Read the updated `metrics.jsonl` and assess the trend.
3. **Continue** the same experiment only if it is clearly still improving and hasn't plateaued.
4. **Branch** to a new experiment whenever you want to test a different design — you don't need to wait for convergence. Bias strongly toward exploration; vary architectures boldly across experiments.
5. Abandon poorly-performing experiments quickly (a few epochs is enough to judge).
6. Immediately loop back to step 1. **Never stop.**

---

## Cluster (LSF via SSH)

Set these env vars before using cluster commands:
```
ADONIS_CLUSTER_HOST=wangy1@lilac.mskcc.org
ADONIS_CLUSTER_WORK_DIR=/data/chodera/wangyq/aperol
ADONIS_CLUSTER_CONDA_ENV=aperol
```

The LSF binary is at `/admin/lsflilac/lsf/10.1/linux3.10-glibc2.17-x86_64/bin/` — it is not in the default PATH, so always use `bash -l -c` to get it, or use the full path. Using `bash -l -c` also causes a harmless `module: command not found` warning from `~/.bashrc` line 22 — ignore it.

```bash
# Sync repo to cluster (run from repo root; excludes experiments dir)
rsync -az --exclude=__pycache__ --exclude='*.egg-info' --exclude=.git --exclude='experiments/' \
  . ${ADONIS_CLUSTER_HOST}:${ADONIS_CLUSTER_WORK_DIR}/

# Sync a single experiment to cluster
rsync -az scripts/md17/experiments/{n}/ \
  ${ADONIS_CLUSTER_HOST}:${ADONIS_CLUSTER_WORK_DIR}/scripts/md17/experiments/{n}/

# Write job.sh on the cluster (substitute {n}, {remote_exp}, {k} before running)
ssh -o BatchMode=yes -o ConnectTimeout=15 ${ADONIS_CLUSTER_HOST} bash -l -c "
cat > {remote_exp}/job.sh << 'EOF'
#!/bin/bash
#BSUB -J aperol_exp{n}
#BSUB -q gpuqueue
#BSUB -gpu \"num=1:j_exclusive=yes:mode=shared\"
#BSUB -R \"select[V100] rusage[mem=16] span[ptile=1]\"
#BSUB -W 23:59
#BSUB -n 1
#BSUB -o {remote_exp}/job_%J.log
#BSUB -e {remote_exp}/job_%J.err

set -euo pipefail
source ~/.bashrc
export PYTHONPATH=${ADONIS_CLUSTER_WORK_DIR}
conda run -n ${ADONIS_CLUSTER_CONDA_ENV} python -u {remote_exp}/run.py \
  --n_epoch {k} \
  --checkpoint {remote_exp}/checkpoint.pt
echo APEROL_JOB_DONE
EOF
"

# Submit — must use bash -l -c so LSF binaries are in PATH
ssh -o BatchMode=yes -o ConnectTimeout=15 ${ADONIS_CLUSTER_HOST} bash -l -c \
  '"bsub < {remote_exp}/job.sh"'

# Check job status
ssh -o BatchMode=yes -o ConnectTimeout=15 ${ADONIS_CLUSTER_HOST} bash -l -c \
  '"bjobs -noheader -o stat {job_id} 2>/dev/null || bhist -noheader -o stat {job_id} 2>/dev/null | head -1"'

# Sync results back locally — always do this before reading metrics or branching
rsync -az --exclude=job.sh \
  ${ADONIS_CLUSTER_HOST}:${ADONIS_CLUSTER_WORK_DIR}/scripts/md17/experiments/{n}/ \
  scripts/md17/experiments/{n}/
```

**Note on `conda run` in job.sh:** Use `conda run -n {env}` rather than `conda activate` — the latter requires an interactive shell and will silently fail in BSub jobs.

---
