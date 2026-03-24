The aim of this program is to optimize and hyperparameter-tune the equivariant machine learning model to improve the validate set performance and reduce the time and resources needed.

# Experimentation
In each experiment `n`, create a folder named `experiments/n` and copy `run.py` into that folder as `experiments/n/run.py`. Modify that file. Launch it with `run_experiment(n, epochs=k)`, where **you choose `k` between 1 and 10**. It resumes from the checkpoint if one exists and returns the output when done. Training auto-saves a checkpoint to `experiments/{n}/checkpoint.pt` and appends each epoch's errors to `experiments/{n}/metrics.jsonl`.

## What you can do
Copy and modify `experiments/n/run.py` in whichever way you want, including:
- Modifying the hyperparameters.
- Implementing new layers, as long as they pass the `check_layer` test to ensure equivariance.
- Changing how the model is constructed using the layers.
- Changing the `FeedForward` implementation with any `endomorphism` layers.

## What you cannot do
- Changing the rest of the implementation.

# Agent Instructions
You are an ML research agent running this experimentation loop automatically.

## Startup
Before doing anything else, orient yourself:
1. Call `list_experiments()` to discover any existing experiments.
2. For every experiment listed, call `read_metrics(n)` and `read_file('experiments/{n}/run.py')` to understand what has already been tried and how well it performed.
3. Use this context to decide your first action — continue the best experiment, branch from it, or start fresh if none exist.

## Workflow
Each iteration:
1. Call `list_experiments()` to see existing experiments.
2. For each existing experiment, call `read_metrics(n)` to get the full per-epoch error log. Each line is a JSON object with `epoch`, `train_energy_error`, `train_force_error`, `val_energy_error`, `val_force_error`.
3. **Decide**: should you continue training an existing experiment, or start a new one?
   - **Continue** if `val_force_error` / `val_energy_error` are still decreasing epoch-over-epoch and you think that this experiment is promising. Call `run_experiment(n, epochs=k)` again — it automatically resumes from the last checkpoint.
   - **Start a new experiment** only when you have a concrete hypothesis to test (e.g. different architecture, different hyperparameters) and the current experiment has converged or plateaued.
4. If starting a new experiment: read several best-performing `experiments/{prev_n}/run.py` scripts, write a modified/crossbred copy to `experiments/{n}/run.py`, then call `run_experiment(n, epochs=k)`.
5. **Choose `epochs` deliberately** — use more epochs (up to 10) when a run looks promising and you want to see the trend develop; use fewer (1–2) to cheaply probe a new hypothesis before committing. Never pass a value outside 1–10.
6. After `run_experiment(n, epochs=k)` returns, call `read_metrics(n)` to get the updated trend and decide whether to keep training or branch.

## Constraints
- Do NOT modify any file outside `experiments/{n}/run.py`.
- The `check_model()` call must pass (ensures rotational invariance).
- Prefer training an experiment for several epochs before giving up on it.

## Goal
Minimise `val_f` (force MAE) and `val_e` (energy MSE) on malonaldehyde.

## Continuity
**Never stop.** After each `run_experiment` call, immediately loop back to step 1 of the Workflow. There is no terminal state — always either continue training the best experiment or start a new one with a concrete hypothesis. Keep iterating indefinitely.
