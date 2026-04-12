The aim of this program is to optimize and hyperparameter-tune the equivariant machine learning model to improve the validate set performance and reduce the time and resources needed.

# Experimentation

All cluster paths are under `/scratch/yqw/aperol/scripts/md17/`. Connect via SSH socket:
```
ssh -o ControlMaster=no -o ControlPath=~/.config/submitter/sockets/trillium.sock \
    -o BatchMode=yes yqw@trillium-gpu.scinet.utoronto.ca '...'
```

In each experiment `n`, create `experiments/{n}/run.py` as a copy of a previous experiment's script (or the base `run.py` if starting fresh), modify it, then submit a job. Training auto-saves a checkpoint to `experiments/{n}/checkpoint.pt` and appends each epoch's errors to `experiments/{n}/metrics.jsonl`. Both energy error and force error should be well below 1.0. The current `run.py` is just a template — it is very far from optimal.

## Running an experiment

**1. Create the experiment directory:**
```bash
ssh ... 'mkdir -p /scratch/yqw/aperol/scripts/md17/experiments/{n}/logs'
```

**2. Copy a base script:**
```bash
# From a previous experiment:
ssh ... 'cp /scratch/yqw/aperol/scripts/md17/experiments/{prev}/run.py \
             /scratch/yqw/aperol/scripts/md17/experiments/{n}/run.py'
# Or from the base template:
ssh ... 'cp /scratch/yqw/aperol/scripts/md17/run.py \
             /scratch/yqw/aperol/scripts/md17/experiments/{n}/run.py'
```

**3. Write your modified `run.py`** via SSH heredoc.

**4. Write `job.sh`** (substitute `{n}` and `{k}` before running):
```bash
ssh ... 'cat > /scratch/yqw/aperol/scripts/md17/experiments/{n}/job.sh << '"'"'EOF'"'"'
#!/bin/bash
#SBATCH --job-name=hnl
#SBATCH --account=aip-yqw
#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONPATH=/scratch/yqw/aperol
export WANDB_MODE=offline
export WANDB_DIR=/scratch/yqw/aperol/scripts/md17/wandb
cd /scratch/yqw/aperol/scripts/md17/experiments/{n}
conda run -n aperol python -u run.py \
  --n_epoch {k} \
  --checkpoint checkpoint.pt
echo APEROL_JOB_DONE
EOF'
```

Choose `k` between 1 and 10. ~20s/epoch on Trillium GPU. Use 1–2 to probe a hypothesis cheaply; use up to 10 when a run looks promising.

**5. Submit:**
```bash
~/Documents/GitHub/submitter/submitter submit-remote trillium \
  /scratch/yqw/aperol/scripts/md17/experiments/{n}/job.sh
```

**Watch live output:**
```bash
~/Documents/GitHub/submitter/submitter watch trillium {job_id}
```

**Read metrics** after a run:
```bash
ssh ... 'cat /scratch/yqw/aperol/scripts/md17/experiments/{n}/metrics.jsonl'
```

## What you can do
Read all experiments. Based on the best-performing script, test design choice hypothesis by implementing new `experiments/n/run.py` in whichever way you want, including:
- **Boldly** changing the way models are constructed from the layers — reorder layers, remove layers, stack the same layer multiple times, mix different layer types, etc.
- Trying radically different architectures: e.g. deeper vs. shallower networks, different message-passing schemes, skip connections, residual blocks, gating mechanisms.
- Modifying the hyperparameters aggressively (learning rate, batch size, hidden dimensions, number of layers, cutoff radius, etc.).
- Implementing new layers, as long as they pass the `check_layer` test to ensure equivariance.
- Changing the `FeedForward` implementation with any `endomorphism` layers.
- Experimenting with different activation functions, normalization strategies, or aggregation schemes.

## What you cannot do
- Changing the rest of the implementation.
- Changing the data split---`n_tr=n_vl=1000` stays true always.

# Agent Instructions
You are an ML research agent running this experimentation loop automatically.

## Startup
Before doing anything else, orient yourself:
1. List existing experiments: `ssh ... 'ls /scratch/yqw/aperol/scripts/md17/experiments/ | sort -n'`
2. For every experiment listed, read its `metrics.jsonl` and `run.py` via SSH to understand what has already been tried and how well it performed.
3. Use this context to decide your first action — continue the best experiment, branch from it, or start fresh if none exist.

## Workflow
Each iteration:
1. List existing experiments on the cluster.
2. For each existing experiment, read `metrics.jsonl` to get the full per-epoch error log. Each line is a JSON object with `epoch`, `train_energy_error`, `train_force_error`, `val_energy_error`, `val_force_error`.
3. **Decide**: should you continue training an existing experiment, or start a new one?
   - **Continue** only if the experiment is clearly still improving and hasn't plateaued.
   - **Start a new experiment** whenever you want to try a different design — you don't need to wait for convergence. Bias toward exploration: if in doubt, branch and try something different. Vary the architecture boldly across experiments (layer types, layer order, depth, width, skip connections, etc.).
4. If starting a new experiment: copy the best-performing script, apply your modifications via SSH heredoc, write `job.sh`, and submit. Never write `experiments/{n}/run.py` from scratch or use Python imports from another experiment.
5. **Choose `k` deliberately** — use more epochs (up to 10) when a run looks promising; use fewer (1–2) to cheaply probe a new hypothesis before committing.
6. After the job completes, read `metrics.jsonl` to get the updated trend and decide whether to keep training or branch.

## Constraints
- Do NOT modify any file outside `experiments/{n}/run.py`.
- The `check_model()` call must pass (ensures rotational invariance).
- Don't linger on poorly-performing experiments — if a run is not improving after a few epochs, abandon it and try something new.

## Goal
Minimise `val_f` (force MAE) and `val_e` (energy MSE) on malonaldehyde.

## Continuity
**Never stop.** After each job completes, immediately loop back to step 1 of the Workflow. There is no terminal state — always either continue training the best experiment or start a new one with a concrete hypothesis. Keep iterating indefinitely.
