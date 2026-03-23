The aim of this program is to optimize and hyperparameter-tune the equivariant machine learning model to improve the validate set performance and reduce the time and resources needed.

# Experimentation
In each experiment `n`, create a folder named `experiments/n` and copy `run.py` into that folder as `experiments/n/run.py`. Modify that file. Run it with `run_experiment(n)`, which trains for exactly **one epoch** and auto-saves a checkpoint. Call `run_experiment(n)` again to train another epoch from where it left off.

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

## Workflow
Each iteration:
1. Call `list_experiments()` to see existing experiments and their progress.
2. **Decide**: should you continue training an existing experiment, or start a new one?
   - **Continue** if the current experiment's `val_f` / `val_e` are still improving epoch-over-epoch. Call `run_experiment(n)` again on the same `n`.
   - **Start a new experiment** only when you have a concrete hypothesis to test (e.g. different architecture, different hyperparameters) and the current experiment has converged or plateaued.
3. If starting a new experiment: read the best-performing `experiments/{prev_n}/run.py`, write a modified copy to `experiments/{n}/run.py`, then call `run_experiment(n)`.
4. After each epoch, analyse the `val_f` / `val_e` trend and decide whether to keep training or branch.

## Constraints
- Do NOT modify any file outside `experiments/{n}/run.py`.
- The `check_model()` call must pass (ensures rotational invariance).
- Prefer training an experiment for several epochs before giving up on it.

## Goal
Minimise `val_f` (force MAE) and `val_e` (energy MSE) on malonaldehyde.
