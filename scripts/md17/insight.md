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

## Exp 4 — Second optimizer reset from Exp 2
- **Result:** val_force=9.72 (epoch 79), train_force=5.70. Ratio=1.70.
- **Best:** val_force=9.53 at epoch 57.
- **Trend:** 11.9(ep0)→9.53(ep57, best)→9.72(ep79). Plateau ~9.5-9.8.
- **Takeaway:** Another ~24% reduction (10.9→9.72). Ratio growing slightly (1.47→1.70) but acceptable.

## Exp 5 — Third optimizer reset from Exp 4
- **Result:** val_force=8.57 (epoch 60, best), final=8.69 (epoch 79). Train=4.65. Ratio=1.84.
- **Config:** Load exp4/checkpoint.pt, fresh Adam LR=1e-5, StepLR(step_size=20, gamma=0.5), 80 epochs.
- **Trend:** 11.9(ep0)→8.57(ep60, best). Each LR decay gave improvement.
- **Takeaway:** Another ~10% reduction (9.53→8.57 best). Ratio growing (1.61→1.84). Best improvement came at epoch 60 after LR decay.

## Exp 6 — Fourth optimizer reset from Exp 5/best_checkpoint
- **Result:** val_force=8.03 (epoch 40, best), final=8.09 (epoch 79). TrainF=4.08. Ratio=1.97.
- **Config:** Load exp5/best_checkpoint.pt (8.57), fresh LR=1e-5, StepLR(step_size=20, gamma=0.5), 80 epochs.
- **Takeaway:** Only ~6% improvement from exp5's 8.57. The 80-epoch schedule doesn't deplete LR fully (only 4 decays). Exp 7 uses 200 epochs for more thorough optimization.

## Exp 7 — Fifth optimizer reset from Exp 6/best_checkpoint
- **Result:** val_force=7.28 (ep77, best), final=7.32 (ep199). Train=3.26. Ratio=2.23. 200 epochs.
- **Config:** Load exp6/best_checkpoint.pt (8.03), fresh LR=1e-5, StepLR(step_size=20, gamma=0.5).
- **Takeaway:** 9.3% improvement from exp6 (8.03→7.28). 200 epochs significantly better than 80-epoch resets.

## Exp 8 — Sixth optimizer reset from Exp 7/best_checkpoint
- **Result:** val_force=6.63 (ep128, best), final=6.64 (ep199). Train=2.54. Ratio=2.61. 200 epochs.
- **Config:** Load exp7/best_checkpoint.pt (7.28), fresh LR=1e-5, StepLR(step_size=20, gamma=0.5).
- **Takeaway:** 8.9% improvement (7.28→6.63). Slow diminishing returns in chain A.

## Chain A summary (no sender broadcast): Exp 1→2→4→5→6→7→8→10
| Exp | From | BestVal | Reset# |
|-----|------|---------|--------|
| 1 | scratch | 14.2 | 0 |
| 2 | exp1 | 10.8 | 1 |
| 4 | exp2 | 9.5 | 2 |
| 5 | exp4 | 8.6 | 3 |
| 6 | exp5 | 8.0 | 4 |
| 7 | exp6 | 7.28 | 5 |
| 8 | exp7 | 6.63 | 6 |
| 10 | exp8 | 6.07 | 7 (chain A restart from exp8) |
| 11 | exp10 | 5.85 | 8 |

Chain A requires ~9% reduction per reset. Target <1.0 needs ~20 more resets. Very slow.

## Chain B (sender+receiver): Exp 3 only — NOT yet continued
| Exp | From | BestVal | Reset# |
|-----|------|---------|--------|
| 3 | scratch | 12.77 | 0 |

**NOTE (CORRECTION):** Exp 9 was **not** chain B. Job 426833 log shows it loaded from
`exp8/best_checkpoint.pt` (chain A). True chain B has NOT been run yet.

## Exp 13 — Bold B+ (sender+receiver + EdgeToPos/Vel + AngleToEdgeMultiChannel), fresh start
- **Result:** val_force=16.69 (ep156, best), final=16.69 (ep199). Train=10.25. Ratio=1.63. 200 epochs.
- **vs chain A exp1:** exp1 reached 14.2 in 90 epochs. B+ reached 16.7 in 200 epochs (worse start).
- **Takeaway:** More complex architecture starts WORSE than chain A baseline. BUT ratio=1.63 (much lower than chain A's 2.52+), suggesting better generalization potential. Will need optimizer resets to see full potential.

## Exp 14 — Chain A reset from exp12 (RUNNING, job 426928)
- **Config:** Load exp12/best_checkpoint.pt (5.77), fresh LR=1e-5, 200 epochs.
- **Trend:** Best=5.62 at epoch 32. Severe plateau since then (5.62-5.90 range, epochs 32-60).
- **Improvement rate:** ~2.6% improvement from exp12's 5.77 — significantly slower than earlier resets (8-9%).
- **Takeaway:** Chain A has severe diminishing returns. Each reset gives less improvement as model reaches its generalization floor with n_tr=1000. Expected final: ~5.4-5.5.

## Exp 14 — Chain A reset from exp12 (DONE)
- **Result:** val_force=5.62 (epoch 32, best), final=5.72 (epoch 199). TrainF=1.77. Ratio=3.18.
- **Takeaway:** Only 2.6% improvement from exp12 (5.77→5.62). Severe diminishing returns.

## Exp 16 — Chain A reset from exp14 (RUNNING, job 426943)
- **Config:** Standard reset from exp14/best (5.62), no noise, 200 epochs.
- **Expected:** ~5.47 (~2.5% improvement given recent rate)

## Exp 15 — B+ reset from exp13 (staged, submit after exp16)
- **Config:** Load exp13/best_checkpoint.pt (16.69), fresh LR=1e-5, 200 epochs.
- **Priority:** Submit FIRST after exp16 finishes (strategic pivot to B+).

## Exp 15 — Chain B+ reset from exp13 (PREPARED)
- **Config:** Load exp13/best_checkpoint.pt (16.69), fresh LR=1e-5, 200 epochs.
- **Expected:** ~10-14 (large improvement from fresh LR on near-converged B+ model)

## SWA Ensemble (diagnostic)
- **Tried:** Averaging weights from exps 8, 10, 11, 12 → val_force=5.92
- **vs best individual (exp 12):** 5.92 > 5.77 → **WORSE**
- **Takeaway:** Chain A checkpoints are in different loss basins; weight averaging doesn't find a better consensus point. Sequential optimizer resets outperform SWA for this setting.

---

## General Lessons (from prior work, carried forward)

1. **DualPairBaseline** (learned position space + raw Cartesian) is the single biggest architectural improvement — halves val_force, eliminates overfitting relative to baseline.
2. **AngleToEdge** (mean 3-body cosine) consistently improves val_force.
3. **VelocityNormToNode** provides useful scalar geometric info.
4. **Layer-level residuals + SiLU** beat Tanh + no residuals.
5. **Full-epoch train & val averaging + model.eval()** gives honest metrics.
6. **Save scheduler state** in checkpoint — otherwise LR restarts cause energy spikes.
7. **Optimizer reset pattern**: load plateau checkpoint, fresh Adam LR=1e-5, StepLR(step_size=20, gamma=0.5) — gives ~5-20% val_force improvement per reset. Use `best_checkpoint.pt` (not `checkpoint.pt`) as `init_from` — starts from the best-seen model, not the potentially-worse final epoch.
7b. **200-epoch resets (2h time limit) greatly outperform 80-epoch resets**: 200 epochs gives 10 full LR decay steps (1e-5→~1e-8), vs 4 steps for 80 epochs. Always prefer `--n_epoch 200 --time=2:00:00` for optimizer-reset experiments.
8. **StepLR(20, 0.5)** for fine-tuning (vs step_size=10 for initial training from scratch).
8b. **Pickle architecture mismatch**: When optimizer-reset run.py's `Layer` class differs from the init_from checkpoint (e.g., one has `NodeToEdgeSenderBroadcast`, the other doesn't), deserialization gives a Layer with missing attributes → `AttributeError` on first forward pass. Always copy run.py from the SAME chain (e.g., copy from exp8, not exp3, when continuing the exp1-chain resets).
8c. **best_checkpoint.pt can be corrupt if a failed job ran first**: If a job FAILS during training, any best_checkpoint.pt it may have written (from a previous architecture) is WRONG. The final `checkpoint.pt` (epoch N) is safer because only successfully-completed jobs update it. If loading from `best_checkpoint.pt` fails, fall back to `checkpoint.pt`.
9. **weight_decay=1e-4** hurts — raised train error without improving val.
10. **CosineAnnealingWarmRestarts from scratch** is unstable.
11. **4-body (dihedral) features** are numerically unstable early in training.
12. **Vel norm² energy readout** always unstable.
13. **Legendre P2/P3 angle features** unstable.
14. **Edge energy readout** creates overfitting.
15. **Chain A has severe diminishing returns**: improvement per reset fell from ~20% early to ~2-3% after 7+ resets. When this happens, switch to B+ chain (lower ratio, more improvement potential) or try `--weight_noise_std 0.005` to break memorization.
16. **New training options in base run.py**: `--clip_grad_norm N` (gradient clipping), `--weight_noise_std N` (perturb weights at reset), `--optimizer adamw` (decoupled weight decay), `--scheduler plateau` (ReduceLROnPlateau).
17. **submitter run-cmd**: avoids raw SSH for cluster commands. Use `submitter run-cmd trillium 'cmd'` instead of `ssh -o ControlPath=... cmd`. All aperol scripts now use submitter throughout.
