# MD17 Malonaldehyde — Insights

## Base Architecture (Exp 1 — all subsequent exps must share this model class)

- **Model**: DualPairBaseline + AngleToEdge + VelocityNormToNode + residuals
- **Size**: node=edge=128 features, pos=vel=16, depth=5
- **Pair baselines**: PairBaseline (learned position space) + CartesianPairBaseline (raw Cartesian)
- **Data**: n_tr=1000, n_vl=1000 (program.md validated defaults — do NOT change)
- **Loss**: 0.01·energy_mse + 0.99·force_mse
- **CRITICAL**: Class definitions must exactly match exp1/run.py for pickle checkpoint loading

---

## Exp 1 — Best-known architecture from scratch (n_tr=n_vl=1000)
- **Status:** Running (Trillium). Fast convergence with n_tr=1000: ~20s/epoch.
- **Config:** DualPairBaseline + AngleToEdge + VelocityNormToNode + residuals, node=128, depth=5, LR=3e-4, StepLR(10, 0.5).
- **Trend (first 5 epochs):** val_force: 760→442→247→150→108 (rapid early drop)

---

## General Lessons (from prior work, carried forward)

1. **DualPairBaseline** (learned position space + raw Cartesian) is the single biggest architectural improvement — halves val_force, eliminates overfitting relative to baseline.
2. **AngleToEdge** (mean 3-body cosine) consistently improves val_force.
3. **VelocityNormToNode** provides useful scalar geometric info.
4. **Layer-level residuals + SiLU** beat Tanh + no residuals.
5. **Full-epoch train & val averaging + model.eval()** gives honest metrics.
6. **Save scheduler state** in checkpoint — otherwise LR restarts cause energy spikes.
7. **Optimizer reset pattern**: load plateau checkpoint, fresh Adam LR=1e-5, StepLR(step_size=20, gamma=0.5) — gives ~5-20% val_force improvement per reset.
8. **StepLR(20, 0.5)** for fine-tuning (vs step_size=10 for initial training from scratch).
9. **weight_decay=1e-4** hurts — raised train error without improving val.
10. **CosineAnnealingWarmRestarts from scratch** is unstable.
11. **4-body (dihedral) features** are numerically unstable early in training.
12. **Vel norm² energy readout** always unstable.
13. **Legendre P2/P3 angle features** unstable.
14. **Edge energy readout** creates overfitting.
