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

