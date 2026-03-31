The aim of this program is to optimize and hyperparameter-tune the equivariant machine learning model to improve the validate set performance and reduce the time and resources needed.

# Experimentation
In each experiment `n`, call `new_experiment(n, source=prev_n)` to create `experiments/n/run.py` as a physical copy of a previous experiment's script (or the base `run.py` if starting fresh). Then modify that file with `write_file` and launch it with `run_experiment(n, epochs=k)`, where **you choose `k` between 1 and 10**. It resumes from the checkpoint if one exists and returns the output when done. Training auto-saves a checkpoint to `experiments/{n}/checkpoint.pt` and appends each epoch's errors to `experiments/{n}/metrics.jsonl`. Note that both energy error and force error should be well below 1.0 so keep trying. The current design in `run.py` is just a template. It is very far from optimal.

## What you can do
Based on existing `run.py`, write new `experiments/n/run.py` in whichever way you want, including:
- **Boldly** changing the way models are constructed from the layers — reorder layers, remove layers, stack the same layer multiple times, mix different layer types, etc.
- Trying radically different architectures: e.g. deeper vs. shallower networks, different message-passing schemes, skip connections, residual blocks, gating mechanisms.
- Modifying the hyperparameters aggressively (learning rate, batch size, hidden dimensions, number of layers, cutoff radius, etc.).
- Implementing new layers, as long as they pass the `check_layer` test to ensure equivariance.
- Changing the `FeedForward` implementation with any `endomorphism` layers.
- Experimenting with different activation functions, normalization strategies, or aggregation schemes.

## What you cannot do
- Changing the rest of the implementation.
- Changing the data split.

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
   - **Continue** only if the experiment is clearly still improving and hasn't plateaued.
   - **Start a new experiment** whenever you want to try a different design — you don't need to wait for convergence. Bias toward exploration: if in doubt, branch and try something different. Vary the architecture boldly across experiments (layer types, layer order, depth, width, skip connections, etc.).
4. If starting a new experiment: call `new_experiment(n, source=prev_n)` to copy the best-performing script, then use `write_file` to apply your modifications, then call `run_experiment(n, epochs=k)`. Never write `experiments/{n}/run.py` from scratch or use Python imports from another experiment.
5. **Choose `epochs` deliberately** — use more epochs (up to 10) when a run looks promising and you want to see the trend develop; use fewer (1–2) to cheaply probe a new hypothesis before committing. Never pass a value outside 1–10.
6. After `run_experiment(n, epochs=k)` returns, call `read_metrics(n)` to get the updated trend and decide whether to keep training or branch.

## Constraints
- Do NOT modify any file outside `experiments/{n}/run.py`.
- The `check_model()` call must pass (ensures rotational invariance).
- Don't linger on poorly-performing experiments — if a run is not improving after a few epochs, abandon it and try something new.

## Goal
Minimise `val_f` (force MAE) and `val_e` (energy MSE) on malonaldehyde.

## Continuity
**Never stop.** After each `run_experiment` call, immediately loop back to step 1 of the Workflow. There is no terminal state — always either continue training the best experiment or start a new one with a concrete hypothesis. Keep iterating indefinitely.
