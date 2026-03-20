The aim of this program is to optimize and hyperparameter-tune the equivariant machine learning model to improve the validate set performance and reduce the time and resources needed.

# Experimentation
In `n`th iteration, create a folder named `scripts/n` and copy `scripts/run.py` into that folder. Modify that file. With the `conda` environment named `aperol` and with `~/Documents/GitHub/aperol/` appended to `$PYTHONPATH`, run the script and observe the learning curve in terms of validation set force and energy errors.

## What you can do
Modify the copied `scripts/n/run.py` in whichever way you want, including:
- Modifying the hyperparameters.
- Implementing new layers, as long as they pass the `check_layer` test to ensure equivariance.
- Changing how the model is constructed using the layers.
- Changing the `FeedForward` implementation with any `endomorphism` layers.

## What you cannot do
- Changing the rest of the implementation.

# Agent Instructions

You are an ML research agent running this experimentation loop automatically.

## Workflow
Each iteration n:
1. Call `list_experiments()` to find the next available n.
2. Read the base script at `scripts/md17/run.py`.
3. Write a modified copy to `scripts/md17/{n}/run.py`.
4. Run it with `run_experiment(n)` and observe the `val_f` / `val_e` learning curve.
5. Analyse results and decide what to change next.

## Constraints
- Do NOT modify any file outside `scripts/md17/{n}/run.py`.
- The `check_model()` call must pass (ensures rotational invariance).
- Keep `n_epoch` small (e.g. 3–10) so each experiment finishes quickly.

## Goal
Minimise `val_f` (force MAE) and `val_e` (energy MSE) on malonaldehyde.

