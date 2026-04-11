# MD17 Malonaldehyde — Insights

## Base Architecture (Exp 1 — all subsequent exps must share this model class)

- **Model**: DualPairBaseline + AngleToEdge + VelocityNormToNode + residuals
- **Size**: node=edge=128 features, pos=vel=16, depth=5
- **Pair baselines**: PairBaseline (learned position space) + CartesianPairBaseline (raw Cartesian)
- **Data**: n_tr=1000, n_vl=1000 (program.md validated defaults — do NOT change)
- **Loss**: 0.01·energy_mse + 0.99·force_mse
- **CRITICAL**: Class definitions must exactly match exp1/run.py for pickle checkpoint loading

---

## Exp 1 — DualPairBaseline from scratch (n_tr=n_vl=1000)
- **Result:** val_force=14.2, train_force=9.71 (epoch 89). Ratio=1.46 (low overfitting!).
- **Config:** DualPairBaseline + AngleToEdge + VelocityNormToNode + residuals, node=128, depth=5, LR=3e-4, StepLR(10, 0.5), 90 epochs.
- **What worked:** Fast early convergence (760→14 in 90 epochs). Low overfitting (val/train=1.46 vs 4.7× before with n_tr=5000).
- **Takeaway:** n_tr=1000 gives much less overfitting than n_tr=5000. val_force plateau at ~14 — needs optimizer reset to push lower.

## Exp 2 — Optimizer reset from Exp 1 + fresh LR=1e-5
- **Result:** val_force=10.91 (epoch 79), train_force=7.21. Ratio=1.51.
- **Config:** Load exp1/checkpoint.pt, fresh Adam LR=1e-5, StepLR(step_size=20, gamma=0.5), 80 epochs.
- **Trend:** val_force 16.2(ep0)→12.2(ep20)→11.04(ep40)→10.81(ep60)→10.91(ep79). Plateau at ~10.8.
- **Takeaway:** Optimizer reset from exp1 gives 24% improvement (14.2→10.9). Ratio stayed ~1.5 (not growing). Good generalization with n_tr=1000.

## Exp 4 — Second optimizer reset from Exp 2 (RUNNING, job 426790)
- **Config:** Load exp2/checkpoint.pt, fresh Adam LR=1e-5, StepLR(step_size=20, gamma=0.5).
- **Expected:** val_force ≈ 7-9 after 80 epochs.

## Exp 3 — Bold: sender+receiver broadcasts from scratch (ready to run)
- **Config:** Same as exp1 but Layer adds NodeToEdgeSenderBroadcast. Trains from scratch.
- **Status:** Prepared, job.sh ready. Run after exp 4 to compare architecture.

## Exp 3 — Bold: sender+receiver broadcasts from scratch
- **Config:** Same as exp1 but Layer adds NodeToEdgeSenderBroadcast alongside NodeToEdgeBroadcast.
- **Hypothesis:** Full sender+receiver per-layer messages enable richer directed message passing.
- **Status:** Prepared, job.sh ready. Run after exp 2.

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
