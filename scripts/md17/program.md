# ML Experimentation Agent — MD17 Malonaldehyde

You are an ML research agent. Your goal is to iteratively optimize an equivariant graph neural network on the MD17 malonaldehyde dataset, minimizing validation force error (`val_force_error`) and validation energy error (`val_energy_error`). Both should be well below 1.0. The base `run.py` is a starting template — it is far from optimal.

All paths below are relative to `scripts/md17/` inside the repo. The repo root is two levels up.

---

## Orientation (do this first, every session)

1. Read `scripts/md17/insight.md` for a summary of what's been learned.
2. List existing experiments on the cluster:
   ```bash
   ssh -o ControlMaster=no -o ControlPath=~/.config/submitter/sockets/trillium.sock \
       -o BatchMode=yes yqw@trillium-gpu.scinet.utoronto.ca \
       'ls /scratch/yqw/aperol/scripts/md17/experiments/ | sort -n'
   ```
3. For each experiment, read `experiments/{n}/metrics.jsonl` and `experiments/{n}/run.py` via SSH (see Cluster section).
4. Check for any running jobs:
   ```bash
   ssh ... 'squeue -u yqw --noheader -o "%i %j %T %M"'
   ```
5. Decide whether to continue the best experiment, branch from it, or start fresh.

---

## Running experiments

**Create a new experiment directory** on the cluster:
```bash
ssh -o ControlMaster=no -o ControlPath=~/.config/submitter/sockets/trillium.sock \
    -o BatchMode=yes yqw@trillium-gpu.scinet.utoronto.ca \
    'mkdir -p /scratch/yqw/aperol/scripts/md17/experiments/{n}'
```

**Write the run.py** via SSH heredoc (substitute `{n}` before running):
```bash
ssh ... 'cat > /scratch/yqw/aperol/scripts/md17/experiments/{n}/run.py << '"'"'EOF'"'"'
<script contents>
EOF'
```

**Write the job.sh** via SSH heredoc (substitute `{n}`, `{k}` before running):
```bash
ssh ... 'cat > /scratch/yqw/aperol/scripts/md17/experiments/{n}/job.sh << '"'"'EOF'"'"'
#!/bin/bash
#SBATCH -J aperol_exp{n}
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --time=59:00
#SBATCH -n 1
#SBATCH -o /scratch/yqw/aperol/scripts/md17/experiments/{n}/job_%j.log
#SBATCH -e /scratch/yqw/aperol/scripts/md17/experiments/{n}/job_%j.err

set -euo pipefail
export PYTHONPATH=/scratch/yqw/aperol
export WANDB_MODE=offline
export WANDB_DIR=/scratch/yqw/aperol/scripts/md17/wandb
conda run -n aperol python -u /scratch/yqw/aperol/scripts/md17/experiments/{n}/run.py \
  --n_epoch {k} \
  --checkpoint /scratch/yqw/aperol/scripts/md17/experiments/{n}/checkpoint.pt
echo APEROL_JOB_DONE
EOF'
```

For optimizer-reset experiments, add `--init_from /scratch/yqw/aperol/scripts/md17/experiments/{prev}/checkpoint.pt` and any extra args (e.g. `--learning_rate 1e-5`).

**Submit the job** using submitter (preferred — handles cluster flags correctly):
```bash
~/Documents/GitHub/submitter/submitter submit-remote trillium \
  /scratch/yqw/aperol/scripts/md17/experiments/{n}/job.sh
```
Or via raw SSH:
```bash
ssh ... 'sbatch /scratch/yqw/aperol/scripts/md17/experiments/{n}/job.sh'
```

Choose `k` based on data size:
- **n_tr=1000** (default): ~20s/epoch on Trillium GPU → use `k=80–100` to fill the 59-minute slot.
- Probe a new hypothesis cheaply with `k=5`; use `k=80` when a run looks promising.
Training resumes from the checkpoint if it exists and appends to `metrics.jsonl` automatically.

**Only one job may be submitted at a time** (debug partition constraint). Always wait for the current job to finish before submitting the next.

**Watch live output** while a job runs:
```bash
~/Documents/GitHub/submitter/submitter watch trillium {job_id}
```

**Read metrics** after each run via SSH:
```bash
ssh ... 'cat /scratch/yqw/aperol/scripts/md17/experiments/{n}/metrics.jsonl'
```

**Check job status**:
```bash
ssh ... 'squeue -j {job_id} --noheader -o "%T %M" 2>/dev/null || \
  sacct -j {job_id} --noheader -o State --parsable2 2>/dev/null | head -1'
```

---

## What you can change (inside `experiments/{n}/run.py` only)

- How the `Layer` class is constructed — reorder, remove, or stack operations; add skip connections, residual blocks, gating; mix different layer types.
- The `Model` class — depth, width, architecture.
- Hyperparameters — learning rate, batch size, weight decay, feature dimensions, cutoff, loss weights, etc.
- The `FeedForward` factory — swap in any `endomorphism` layers (`LazySquareLinear`, `LazyLayerNorm`, etc.) or activation functions.
- Add entirely new architectures, as long as `check_model()` passes (it verifies rotational equivariance). Call `model.eval()` before `check_model()` if the model has stochastic layers.

**Do not** modify anything outside `experiments/{n}/run.py`. Do not change the data split.

**Do not** change `n_vl` or `n_tr`. The validated defaults (`n_vl=1000`, `n_tr=1000`) must be preserved across all architecture experiments — changing them confounds comparisons.

**Be bold with architecture.** Past wins came from non-obvious structural changes (angle features, residuals, NequIP-style tensor products). When designing new experiments, lean toward high-variance ideas rather than small hyperparameter tweaks. Good candidates include: angle/dihedral features (3-body, 4-body), attention mechanisms, message-passing depth, multi-scale aggregation, tensor product layers, and combined best-of approaches. A bold experiment that fails fast is more informative than a cautious one that barely moves the needle.

---

## Insights log

Keep a running record of learned experiences in `scripts/md17/insight.md` (local copy; also at `/scratch/yqw/aperol/scripts/md17/insight.md` on the cluster). After every experiment (or whenever you draw a meaningful conclusion), append an entry in this format:

```markdown
## Exp {n} — <one-line description>
- **Result:** val_force_error=X.XX, val_energy_error=X.XX (epoch K)
- **What worked / didn't:** ...
- **Takeaway:** ...
```

Read `insight.md` at the start of each session (during Orientation) so prior findings inform new designs. Never overwrite the file — only append.

---

## Workflow (iterate indefinitely)

After each run completes:
1. **Read metrics** via SSH. Assess the trend — is it still improving?
2. **Append to insight.md** with the result and takeaway.
3. **Continue** the same experiment (submit another job) only if it is clearly still improving and hasn't plateaued.
4. **Branch** to a new experiment whenever you want to test a different design. Run experiments sequentially — never submit a new job until the current one has finished. Bias strongly toward exploration; vary architectures boldly across experiments.
5. Abandon poorly-performing experiments quickly (a few epochs is enough to judge).
6. Immediately loop back to step 1. **Never stop.**

---

## Cluster (SLURM on Trillium/SciNet via submitter)

All cluster interaction goes through SSH master sockets managed by the `submitter` tool. The socket must already be open (run `submitter connect` interactively first if needed).

```
Submitter:   ~/Documents/GitHub/submitter/submitter
Config:      ~/Documents/GitHub/submitter/clusters.conf
Sockets:     ~/.config/submitter/sockets/
Cluster:     trillium  →  yqw@trillium-gpu.scinet.utoronto.ca
Remote dir:  /scratch/yqw/aperol
Conda env:   aperol
```

**SSH shorthand** (reuse existing master socket, no MFA):
```bash
ssh -o ControlMaster=no \
    -o ControlPath=~/.config/submitter/sockets/trillium.sock \
    -o BatchMode=yes \
    yqw@trillium-gpu.scinet.utoronto.ca \
    '<remote command>'
```

**Submitter commands:**
```bash
submitter status                               # Check which clusters are connected
submitter connect                              # Open master connections (interactive, MFA required)
submitter submit-remote trillium <remote-path> # Submit a job script already on the cluster
submitter poll trillium <jobid>                # Wait for job to finish; exit 0 if COMPLETED
submitter cancel trillium <jobid>              # Cancel a running/pending job
submitter watch trillium <jobid>               # Tail job stdout live (Ctrl+C to stop)
submitter fetch trillium <jobid>               # Copy job log files to current directory
submitter jobs trillium                        # Show recent jobs (last 24h)
```

**Note on the `debug` partition:** Only one job at a time (no pending jobs allowed — QOS limit). Always wait for the current job to complete before submitting the next. `submitter chain` does NOT work here (it would require a pending slot). Use `submitter poll` + `submit-remote` in a script, or submit manually.

**Note on `conda run` in job.sh:** Use `conda run -n {env}` rather than `conda activate` — the latter requires an interactive shell and will silently fail in SLURM jobs.

**Note on heredoc quoting over SSH:** To write multi-line scripts remotely, wrap the heredoc delimiter in single quotes so the local shell doesn't expand variables:
```bash
ssh ... 'cat > remote/path/file.py << '"'"'EOF'"'"'
content with $variables preserved literally
EOF'
```
