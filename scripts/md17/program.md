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

**Run an experiment** by submitting a SLURM job (see the Cluster section). Choose `k` between 1 and 10. Use fewer epochs (1–2) to cheaply probe a new hypothesis; use more (up to 10) when a run looks promising. Training resumes from the checkpoint if it exists and appends to `metrics.jsonl` automatically.

**Only one job may be submitted at a time** (debug partition constraint). Always wait for the current job to finish before submitting the next.

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
1. **Preserve copies.** Immediately after a run completes, ensure these two files exist:
   - `scripts/md17/experiments/{n}/run.py` — the exact script that was run
   - `scripts/md17/experiments/{n}/metrics.jsonl` — the full epoch log
   These are written directly to the work directory on the cluster.
2. Read the updated `metrics.jsonl` and assess the trend.
3. **Continue** the same experiment only if it is clearly still improving and hasn't plateaued.
4. **Branch** to a new experiment whenever you want to test a different design. Run experiments sequentially — never submit a new job until the current one has finished. Bias strongly toward exploration; vary architectures boldly across experiments.
5. Abandon poorly-performing experiments quickly (a few epochs is enough to judge).
6. Immediately loop back to step 1. **Never stop.**

---

## Cluster (SLURM on Trillium/SciNet)

Running directly on the Trillium login node — no SSH needed. Set these env vars before using cluster commands:
```
ADONIS_CLUSTER_WORK_DIR=/scratch/yqw/aperol
ADONIS_CLUSTER_CONDA_ENV=aperol
```

```bash
# Write job.sh (substitute {n}, {k} before running)
cat > ${ADONIS_CLUSTER_WORK_DIR}/scripts/md17/experiments/{n}/job.sh << 'EOF'
#!/bin/bash
#SBATCH -J aperol_exp{n}
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=23:59:00
#SBATCH -n 1
#SBATCH -o /scratch/yqw/aperol/scripts/md17/experiments/{n}/job_%j.log
#SBATCH -e /scratch/yqw/aperol/scripts/md17/experiments/{n}/job_%j.err

set -euo pipefail
export PYTHONPATH=/scratch/yqw/aperol
conda run -n aperol python -u /scratch/yqw/aperol/scripts/md17/experiments/{n}/run.py \
  --n_epoch {k} \
  --checkpoint /scratch/yqw/aperol/scripts/md17/experiments/{n}/checkpoint.pt
echo APEROL_JOB_DONE
EOF

# Submit
sbatch ${ADONIS_CLUSTER_WORK_DIR}/scripts/md17/experiments/{n}/job.sh

# Check job status
squeue -j {job_id} --noheader -o "%T" 2>/dev/null || \
  sacct -j {job_id} --noheader -o State --parsable2 2>/dev/null | head -1
```

**Note on the `debug` partition:** Only one job can run at a time. Never submit a new job while one is already queued or running — always wait for the current job to finish first.

**Note on `conda run` in job.sh:** Use `conda run -n {env}` rather than `conda activate` — the latter requires an interactive shell and will silently fail in SLURM jobs.

---
