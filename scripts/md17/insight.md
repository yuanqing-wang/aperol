# MD17 Malonaldehyde — Insights (Trillium series)

## Base Architecture (Exp 1 — all subsequent exps share this model class)

- **Model**: DualPairBaseline + AngleToEdge + VelocityNormToNode + residuals
- **Size**: node=edge=128 features, pos=vel=16, depth=5
- **Pair baselines**: PairBaseline (learned position space) + CartesianPairBaseline (raw Cartesian)
- **Data**: n_tr=1000, n_vl=1000 (val at indices [5000,6000))
- **Loss**: 0.01·energy_mse + 0.99·force_mse
- **CRITICAL**: Class definitions must exactly match exp1/run.py for pickle checkpoint loading

---

## Optimizer Reset Pattern (Exps 3, 5–16)

Each reset: load previous plateau's checkpoint, fresh Adam LR=1e-5, StepLR(step_size=20, gamma=0.5), run ~90 epochs until LR depletes to 6.25e-7 and val_force plateaus.

| Exp | Description | Best val_force | Train_force at plateau |
|-----|-------------|---------------|----------------------|
| 1 | Initial training (LR=3e-4, StepLR(10,0.5)) | 1.138 (ep83) | — |
| 2 | Context-conditioned Cartesian pair (from exp1) | 1.165 — **WORSE, abandoned** | — |
| 3 | Optimizer reset from exp1, fresh LR=5e-5 | 0.865 (ep115) | ~0.534 |
| 4 | From exp3, weight_decay=1e-4 | 0.896 — **WORSE** (raised train, hurt val) | — |
| 5 | 2nd reset from exp3, LR=1e-5 | 0.721 (ep85) | — |
| 6 | 3rd reset from exp5 | 0.614 (ep96) | — |
| 7 | 4th reset from exp6 | 0.546 (ep99) | — |
| 8 | 5th reset from exp7 | 0.493 (ep82) | — |
| 9 | 6th reset from exp8 | 0.453 (ep82) | — |
| 10 | 7th reset from exp9 | 0.419 (ep87) | ~0.130 |
| 11 | 8th reset from exp10 | 0.394 (ep87) | ~0.112 |
| 12 | 9th reset from exp11 | 0.370 (ep87) | ~0.098 |
| 13 | 10th reset from exp12 | 0.354 (ep87) | ~0.087 |
| 14 | 11th reset from exp13 | 0.337 (ep87) | ~0.078 |
| 15 | 12th reset from exp14 | **0.324** (ep85) | ~0.069 |
| 16 | 13th reset from exp15 — **RUNNING** | — | — |

**Overfitting ratio** at exp15: val/train = 0.324/0.069 = **4.7×** — severe and growing.

---

## Key Findings

### What worked
1. **Optimizer reset pattern** is the single most effective technique — each fresh LR=1e-5 from a depleted checkpoint gives ~4-22% improvement.
2. **DualPairBaseline** (learned position + raw Cartesian) provides strong inductive bias and fast convergence (vs single pair in the old series).
3. **AngleToEdge** — mean 3-body cosine consistently helps.
4. **VelocityNormToNode** — useful scalar geometric info.
5. **Layer-level residuals + SiLU** — essential for deep networks.
6. **Full-epoch val averaging + model.eval()** — honest metrics.
7. **StepLR(step_size=20, gamma=0.5)** for fine-tuning (vs step_size=10 for initial training).

### What didn't work
1. **Context-conditioned Cartesian pair** (exp2): worse than simple pair potential.
2. **weight_decay=1e-4** (exp4): raised train error without improving val.
3. **Higher LR on reset** (exp3 used 5e-5; subsequent resets use 1e-5 to be safe after seeing exp4 instability).

### Diminishing returns pattern
Improvement per reset is shrinking: 22%→19%→15%→11%→10%→8%→7.5%→6%→6%→4.5%→4.8%→3.9%→?

---

## Current Status

- **Exp 16** running on Trillium (job 426611). 13th optimizer reset from exp15.
- Expected val_force ≈ 0.310–0.318 when plateaus.
- **Main concern**: overfitting ratio 4.7× and growing — generalization is the bottleneck.

## Next Experiment Ideas

- **Exp 17**: Continue optimizer reset pattern (safe, ~4% gain expected).
- **Exp 17-alt**: Try SWA (Stochastic Weight Averaging) from exp16 checkpoint — may close the train/val gap.
- **Do NOT retry**: weight_decay alone (exp4 showed it hurts), raw Cartesian pair without learned pair, higher LR resets from near-converged checkpoints.
