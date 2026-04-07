# MD17 Malonaldehyde — Insights

## Architecture & Data

- **Energy targets are pre-normalized** (zero mean, unit std over full dataset before split). So val_energy_error (MSE) < 1.0 means energy predictions within 1 std — achievable. Forces are NOT normalized and remain in kcal/mol/Å.
- **Force MSE target** (<1.0) corresponds to RMSE < 1.0 kcal/mol/Å — SOTA methods (NequIP, PaiNN) achieve ~0.25–0.3 MAE, so this is achievable but requires a capable equivariant model and sufficient training epochs.
- **PyTorch 2.6 change**: `torch.load` defaults to `weights_only=True`, which breaks checkpoint loading of model classes. Always use `weights_only=False`.

---

## Exp 0 — Wider + deeper + SiLU + residuals + VelocityNormToNode
- **Result:** val_force=49.8, val_energy=15.1 (epoch 14)
- **What worked:** Bigger model (32 features, depth 3), SiLU FeedForward (no saturating Tanh), layer-level residuals, VelocityNormToNode.
- **What didn't:** LR=5e-4 too high (energy error spiked to 44 at epoch 14). Scheduler rebuilt from scratch on resume → instability.
- **Takeaway:** Residuals + SiLU + VelocityNormToNode are all worth keeping. Need scheduler state saving + lower/stable LR.

---

## Exp 1 — Angle/3-body features + linear warmup
- **Result:** val_force=26.2, val_energy=17.5 (epoch 14, best so far)
- **What worked:** `AngleToEdge` module (mean cosine of 3-body angles injected into edges) improved beyond exp 0. LR=3e-4 with warmup.
- **What didn't:** Scheduler restarts from scratch on resume → catastrophic spike at epoch 12 (train_energy=1015). val metrics from single-batch — not reliable.
- **Takeaway:** Angle features help. Scheduler state must be saved. Single-batch val is unreliable.

---

## Exp 2 — MAE loss + per-atom energy normalization
- **Result:** Failure — val_force_mse=6779 even after 10 epochs
- **What didn't:** Missing `model.eval()` in validation caused inconsistent statistics. Per-atom energy normalization decoupled loss scale from force gradients. MAE (train) vs MSE (val) created uninterpretable metrics.
- **Takeaway:** Always call `model.eval()` before validation. Keep loss and metrics consistent (MSE throughout). No per-atom normalization needed since dataset already normalizes energy.

---

## Exp 3 — Clean training: full-epoch avg, model.eval(), saved scheduler, LR=1e-4
- **Result:** val_force=29.0 (epoch 19); still converging
- **What worked:** Full-epoch running average for train metrics (honest). model.eval() during validation. Saved/restored scheduler state. StepLR(10, 0.5) decays LR at epoch 10.
- **What didn't:** LR=1e-4 slower than exp 1's LR=3e-4. Energy errors still high (14–65 MSE).
- **Takeaway:** Full-epoch averaging and model.eval() are critical for honest metrics. Model is NOT overfitting (train ≈ val), needs more training/capacity.

---

## Exp 4 — Same as exp 3 but LR=3e-4
- **Result:** val_force=38.8 (epoch 9); similar trajectory to exp 1 but with honest metrics
- **Trend:** 352→247→106→91→71→51→51→42→40→39 — steady convergence
- **What worked:** Higher LR (3e-4) converges faster. Wider model (64 features, depth 5).
- **Takeaway:** LR=3e-4 with StepLR(10, 0.5) is the best stable schedule found so far. Continue for 30+ more epochs.

---

## Exp 5 — Wider model: 96 features, depth 5
- **Result:** val_force worse than exp 4 at same epochs
- **What didn't:** More capacity did not help; likely needs more data or better architecture, not just width.
- **Takeaway:** Wider (96 features) is not obviously better than 64 features at this data scale. Capacity is not the bottleneck.

---

## Exp 6/7 — Higher energy_weight (0.1, 0.5)
- **Result:** Forces got significantly worse (higher val_force MSE)
- **What didn't:** Increasing energy weight to fix energy underdetermination hurts force quality, which is the primary metric.
- **Takeaway:** The energy vs force trade-off is real. Keep energy_weight=0.01 (or lower) for best forces.

---

## Exp 8 — Two-phase training (forces first, then energy)
- **Result:** Slow force convergence; energy phase didn't add much
- **What didn't:** Two-phase schedule is complicated and doesn't improve forces vs single-phase.
- **Takeaway:** Single-phase training with energy_weight=0.01 is simpler and at least as good.

---

## Exp 9 — Batch-centered energy difference loss
- **Result:** CATASTROPHIC FAILURE — val_force=1097
- **What didn't:** Centering energy predictions within a batch introduces cross-sample gradient dependencies that destabilize force gradients via create_graph=True.
- **Takeaway:** Never use batch-level statistics inside the energy loss when forces use create_graph=True.

---

## Exp 10 — Linear energy calibration diagnostic
- **Result:** val_energy_cal_error=0.26–0.60 from epoch 0! val_force still ~5.7
- **BREAKTHROUGH:** Energy is already well-predicted up to a constant offset (the linear calibration removes it). MSE(a*E_pred+b, E_true) is already < 1.0 from the start. The ONLY remaining challenge is forces.
- **What worked:** Post-hoc OLS calibration: fit (a, b) on training set, apply to val predictions. Reports true energy quality without needing to change loss.
- **Takeaway:** Energy is NOT the problem. Stop worrying about energy. Focus entirely on reducing val_force_mse from ~5.7 to < 1.0.

---

## Exp 11 — Reset optimizer from exp 4 checkpoint (fresh LR=1e-5)
- **Result:** val_force=3.70, val_energy_cal=0.026 (epoch 99) — BEST SO FAR
- **What worked:** Fresh Adam at LR=1e-5 from exp 4's frozen checkpoint. Model trained from ~5.7→3.7 in 100 epochs.
- **Key observation:** train_force=1.51, val_force=3.70 — generalization gap of 2.2. Model is slightly overfitting.
- **Takeaway:** Resetting optimizer unblocks frozen training. val_force <3.7 needs better architecture or regularization. Cannot load a different architecture as init_checkpoint because loaded objects use old class definitions.

---

## Exp 12 — CrossProductToVelocity + CosineAnnealingWarmRestarts (fresh)
- **Result:** val_force oscillating 32-85, did not converge — CANCELLED
- **What didn't:** CosineAnnealingWarmRestarts with LR=3e-4 caused severe oscillation during LR restarts (epoch 20). Too aggressive.
- **Takeaway:** High LR + cosine restart on fresh models is destabilizing. Use StepLR for stable training.

---

## Exp 13 — From exp 4, LR=1e-4 + CosineAnnealingWarmRestarts
- **Result:** val_force oscillating 4.2-5.1 — CANCELLED
- **What didn't:** LR=1e-4 too high for fine-tuning a near-converged model. Noisy, no improvement over exp 11.
- **Takeaway:** Fine-tuning from a pre-trained checkpoint requires small LR (1e-5 as in exp 11). 1e-4 causes chaos.

---

## Exp 14 — depth=8 from scratch + CrossProductToVelocity + StepLR(25, 0.7)
- **Result:** val_force=4.8-5.5 at epoch 97 — CANCELLED (worse than exp 4 depth=5)
- **What didn't:** Deeper model converged more slowly and to a WORSE floor than depth=5. depth is NOT the bottleneck.
- **Takeaway:** Going deeper doesn't help if the representational capacity isn't the bottleneck.

---

## Exp 15 — L1 force loss from scratch
- **Result:** val_force_MSE=10.2 at epoch 66 — CANCELLED (much worse)
- **What didn't:** Training with L1 loss doesn't optimize MSE. val_force_mse=10.2 vs train_l1=1.75 (apples/oranges).
- **Takeaway:** Always optimize the same metric you're measuring. L1 training → L1 val metric, not MSE.

---

## Exp 16 — Dropout(p=0.1) in FeedForward
- **Result:** FAILED — equivariance check broke
- **What didn't:** check_model runs two forward passes; different dropout masks → different results → equivariance test fails.
- **Fix:** Call model.eval() before check_model. Resubmitted.
- **Takeaway:** Dropout breaks check_model unless model is in eval mode during the test.

---

## Exp 17 — 4-body DihedralToEdge
- **Result:** FAILED — val_force exploded to 950-1380 in first 3 epochs, CANCELLED
- **What didn't:** DihedralToEdge computes O(N⁴) cross products. Even with correct batch/unbatched negative-index unsqueeze, the 4-body term is numerically unstable early in training (gradient magnitudes explode).
- **Takeaway:** 4-body features are too unstable. Skip them unless there's a warm-start from a converged checkpoint.

---

## Exp 18 — Vel norm² readout + CrossProductToVelocity
- **Result:** val_force oscillating 60-103 at epoch 29, CANCELLED (much worse than baseline)
- **What didn't:** Adding velocity norm² to the energy readout (alongside node features) plus CrossProductToVelocity made training unstable and slow.
- **Takeaway:** Modifying the readout to include equivariant features via vel_norm² doesn't help and creates instability. Keep ProjectionOut simple.

---

## Exp 20 — Wider equivariant channels (position_features=32, velocity_features=32)
- **Result:** SEVERE OVERFITTING — train_force=5.75, val_force=213 at epoch 45, CANCELLED
- **What didn't:** Doubling the equivariant channel capacity from 16 to 32 caused catastrophic overfitting. The model memorized training geometry without generalizing.
- **Takeaway:** More equivariant channel capacity = more overfitting with 1000 training samples. Wider equivariant channels are NOT beneficial at this data scale.

---

## Exp 21 — Legendre polynomial angle expansion (P1+P2+P3)
- **Result:** FAILED — val_force exploded (955, 1379, 748 in first 3 epochs), CANCELLED
- **What didn't:** P3 has 6× larger gradient than P1. Concatenating P1+P2+P3 with a single weight matrix changes the initialization scale, destabilizing early training.
- **Takeaway:** Legendre polynomials P2 and P3 are unstable as concatenated edge features. If using higher-order Legendre, use separate weight matrices per order.

---

## Exp 22 — Pairwise energy baseline (direct gradient path via pair potential)
- **Result (ongoing):** val_force: 25.7 (ep10) → 9.82 (ep27) → 8.03 (ep33) → 8.18 (ep35). BEST SO FAR, still converging.
- **BREAKTHROUGH:** Adding a learned pair potential E_pair = sum_{ij} MLP(RBF(d²_ij)) in LEARNED position feature space provides a direct, clean gradient path from energy to positions.
- **What worked:** Near-zero overfitting (train≈val the entire time). At epoch 22, ratio is only 1.15 vs exp 11's final ratio of 2.45. Converges 2× faster than baseline.
- **Key mechanism:** The pair potential provides a physics-informed prior: forces are related to pairwise distances. This strong inductive bias enables rapid convergence and good generalization.
- **Takeaway:** Pairwise energy baselines in LEARNED position feature space are transformative. The pair potential in the message-passing position space (not raw Cartesian) is better than raw Cartesian pair potentials (exp 25 failed).

---

## Exp 23 — Stronger weight decay (1e-5)
- **Result:** val_force=14.6 at epoch 25. Worse than exp 22 at same stage.
- **What didn't:** weight_decay=1e-5 alone doesn't dramatically improve over baseline. Convergence is slower than exp 22's pair baseline.
- **Takeaway:** Weight decay alone is insufficient. The pairwise baseline (exp 22) is far superior for addressing overfitting.

---

## Exp 24 — P2 Legendre angle + pairwise baseline
- **Result:** val_force oscillating 50-85 at epoch 5, CANCELLED (worse than exp 22 alone)
- **What didn't:** The P2 angle expansion caused oscillation even with separate weights (possibly the combined gradient complexity).
- **Takeaway:** P2 angle features destabilize training even with pairwise baseline. The baseline alone (exp 22) is cleaner and better.

---

## Exp 25 — Atom-typed pair potential on raw Cartesian positions
- **Result:** val_force went from 60→97 (getting WORSE at epoch 4), CANCELLED
- **What didn't:** Raw Cartesian pair potential is inferior to learned-position pair potential. The learned position features encode more geometry information.
- **Takeaway:** Using `state.position` (learned, after message passing) for pair potential is better than using `sample.position` (raw Cartesian). The message-passing transformation encodes geometry more effectively.

---

## Exp 16 — Dropout (p=0.1) in FeedForward
- **Result:** Plateaued at val_force=34 for 20+ epochs, CANCELLED (epoch 52)
- **What didn't:** Dropout did not help with overfitting in a meaningful way. The model plateaued at 34, far behind exp 22's 10.9 at similar epochs.
- **Takeaway:** Dropout is not effective for this problem. The pairwise baseline (exp 22) is far superior for addressing overfitting.

---

## Exp 26 — Multi-scale pair baselines (after layer 2 and layer 5)
- **Result:** val_force=14.3 at epoch 22. Worse than exp 22 (which has ~10.9 at epoch 22).
- **What didn't:** Adding a mid-network pair baseline didn't help. The intermediate state.position after only 2 layers encodes less useful geometry than the final 5-layer position.
- **Takeaway:** Single pair baseline at the final layer is better than multi-scale. Keep it simple.

---

## Exp 27 — Per-channel pair distances (Dx×n_rbf input instead of mean)
- **Result:** val_force=41.4 at epoch 8. Much slower convergence than exp 22.
- **What didn't:** Using all 16 channel distances separately (160 input) instead of mean (20 input) is harder to optimize. The mean is a better inductive bias — it's a single radial distance which is physically meaningful.
- **Takeaway:** Mean-distance pair potential (exp 22) is better than per-channel pair potential. The mean acts as a learned norm in position feature space, which is the right invariant.

---

## Exp 28 — Pair + Triplet energy baselines
- **Status:** Initially crashed (TripletEnergyBaseline.expand_as bug). Fixed with broadcast_tensors. Resubmitted.
- **Fix:** Changed `expand_as(rbf_jk)` to `torch.broadcast_tensors(rbf_ij, rbf_ik, rbf_jk)` before torch.cat. This handles both batched and unbatched shapes.

---

## Exp 29 — Node + Edge + Pairwise energy readout
- **Status:** Submitted. Tests whether adding edge features to the energy readout (alongside node + pair) helps via additional gradient paths through edges.

---

## Exp 30 — More training data (n_tr=5000) with pair baseline
- **Status:** Submitted. Tests whether data quantity is the bottleneck (since overfitting is already solved by pair baseline).

---

## Exp 31 — Atom-type conditioned pair potential
- **Status:** Submitted. MLP(RBF(d²_ij) ⊕ type_i ⊕ type_j) learns separate potentials per pair type (C-H ≠ C-C ≠ O-H). Uses LazyLinear for atom-type-independent initialization.

---

## General Lessons

1. **Angle features (3-body) help** — AngleToEdge consistently improves val_force.
2. **Residuals + SiLU** beat Tanh + no residuals from the base template.
3. **VelocityNormToNode** provides useful scalar geometric info and should always be included.
4. **model.eval() in validation is mandatory** — missing it corrupts val metrics.
5. **Full-epoch train averaging** gives more honest signal than last-batch.
6. **Save scheduler state** in checkpoint — otherwise LR restarts cause energy spikes.
7. **Model is underfitting** (train ≈ val force error at epoch 9 for exp 4) — need more epochs and/or more capacity.
8. **StepLR(10, 0.5)** is simpler and more resume-stable than cosine/warmup schedules.
