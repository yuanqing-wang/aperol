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
- **Result (FINAL):** val_force: 25.7 (ep10) → 9.82 (ep27) → 8.03 (ep33) → 6.04 (ep73) → 5.95 (ep100). CANCELLED at ep100 — plateaued. Superseded by 5000-sample experiments.
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
- **Result:** val_force=41.4 at epoch 8, CANCELLED (too slow)
- **What didn't:** Using all 16 channel distances separately (160 input) instead of mean (20 input) is harder to optimize. The mean is a better inductive bias — it's a single radial distance which is physically meaningful.
- **Takeaway:** Mean-distance pair potential (exp 22) is better than per-channel pair potential. The mean acts as a learned norm in position feature space, which is the right invariant.

---

## Exp 28 — Pair + Triplet energy baselines
- **Result:** train_force=491 at epoch 10, CATASTROPHIC, CANCELLED
- **What didn't:** The O(N³) triplet energy term explodes even after fixing broadcast_tensors bug. Gradient magnitude from triplet term is too large.
- **Takeaway:** Triplet energy baselines in learned position space are unstable. Not worth pursuing without careful initialization or weight scaling.

---

## Exp 29 — Node + Edge + Pairwise energy readout
- **Result:** val_force=241, train_force=98 at epoch 9. val >> train (overfitting!), CANCELLED
- **What didn't:** Adding edge energy readout to node+pair readout actually INCREASED overfitting. Edge features may be more dataset-specific than position features.
- **Takeaway:** Edge energy readout creates overfitting despite the pair baseline. Node+pair readout (exp 22) is sufficient.

---

## Exp 30 — More training data (n_tr=5000) with pair baseline
- **Status:** Running. val_force: 37.8 (ep1) → 15.1 (ep5) → 9.21 (ep6). 5× faster convergence than exp 22 (same n_gradient_steps). Different val set (indices [5000, 6000)).
- **Insight:** 5000 samples dramatically improves convergence rate. Each epoch = 625 batches vs exp 22's 125 batches — 5× more gradient steps per epoch.

---

## Exp 31 — Atom-type conditioned pair potential
- **Result:** train_force=8274 at epoch 2, CATASTROPHIC EXPLOSION, CANCELLED
- **What didn't:** Concatenating atom type one-hot vectors to the pair MLP input caused immediate gradient explosion.
- **Takeaway:** Atom-type conditioning of the pair potential is unstable. The atom type information already flows through message passing (node features start from atom_type). Don't inject it again at the pair potential level.

---

## Exp 32 — Dual pair potential (raw Cartesian + learned position feature distances)
- **Result (FINAL):** val_force: 11.7 (ep28) → 10.0 (ep32) → 8.95 (ep55). CANCELLED — with 1000 samples, dual pair eventually overfits (train=4.37 vs val=8.95 at ep55, ratio=2.05). Much worse than exp 39 (dual pair + 5000 samples).
- **Takeaway:** Dual pair + 1000 samples eventually overfits just like standard pair + 1000 samples. The 5000-sample versions (exp 38, 39) avoid this via more data.

---

## Exp 33 — Velocity norm² energy readout + pairwise baseline
- **Result:** train_force=116971 at epoch 0, CATASTROPHIC EXPLOSION, CANCELLED
- **What didn't:** Velocity norm² energy readout explodes immediately (same as exp 18). Should not have been attempted again — exp 18 already established this fails.
- **Takeaway:** Vel norm² energy readout is always unstable. Do not retry. Read insight.md before designing new experiments.

---

## Exp 34 — 10000 training samples with pair baseline
- **Result:** CANCELLED at ep2. val_force=309 (ep0), 283 (ep1), 236 (ep2) — very slow convergence.
- **What didn't:** 10000 samples with 1250 batches/epoch converges MUCH slower than 5000 samples. At ep2 (2500 gradient steps), exp 34 has val_force=236 vs exp 30 at ep4 (2500 steps) with val_force=15.9. Something is fundamentally wrong.
- **Hypothesis:** Larger dataset with same batch_size=8 may result in poorer gradient coverage per epoch due to less data recycling. Or the train average is dominated by very early batches when LR was not yet effective.
- **Takeaway:** More training data does NOT automatically improve convergence rate. 5000 samples is a sweet spot — more data may need larger batch size or different LR scaling.

---

## Exp 36 — Wider model (node=128) with 5000 training samples
- **Status:** Running. val_force: 16.3 (ep3) → 13.5 (ep5) → 6.97 (ep6) → 4.32 (ep10). NEW OVERALL BEST!
- **BREAKTHROUGH:** At epoch 10 post-LR-decay (3e-4 → 1.5e-4), exp 36 jumped to val_force=4.32 with near-zero overfitting (val/train = 4.32/4.47 = 0.97). This surpasses exp 22's 5.97 after 89 epochs.
- **Key mechanism:** LR decay at epoch 10 triggers large convergence jump. Node=128 + 5000 samples is the sweet spot — wider model can represent the PES better without overfitting.
- **Takeaway:** node=128 + 5000 samples + standard pair baseline is the best configuration found so far. The near-zero overfitting means continued improvement is expected.

---

## Exp 37 — Deeper model (depth=7) with 5000 training samples
- **Status:** Running. val_force=63.5 (ep0), 47.0 (ep1). Too early to judge.

---

## Exp 38 — DualPairBaseline with 5000 training samples
- **Status:** Running. val_force=65.9 (ep0), 33.9 (ep1). Early, needs more epochs.

---

## Exp 39 — DualPairBaseline + node=128 + 5000 training samples
- **Status:** Running. val_force: 15.0 (ep4) → 6.94 (ep8) → 5.51 (ep10) → 3.14 (ep11) → **2.83 (ep14)**. NEW BEST.
- **BREAKTHROUGH:** Post-LR-decay at epoch 10: 5.51 → 3.14 in one epoch. By ep14 = 2.83, with val < train (0.97) — no overfitting. Projected to reach <1.0 around epoch 20-25 (next LR decay at 7.5e-5).
- **Key observation:** Dual pair + node=64 (exp 38) converges MUCH slower (ep8=15.30) than dual pair + node=128 (exp 39, ep8=6.94). The wider model benefits much more from the dual pair baseline.
- **Takeaway:** Dual pair + node=128 + 5000 samples is the best configuration found. Near-zero overfitting enables continued improvement at each LR decay.

---

## Exp 41 — Deeper (depth=7) + DualPairBaseline + 5000 samples
- **Result:** CANCELLED at ep1. val_force=133 at ep1 vs exp 37 (depth=7, standard pair)=47 at ep1. Depth=7 + DualPair combination is much harder to optimize than either alone.
- **Takeaway:** Combining depth=7 with DualPairBaseline creates optimization difficulties. The dual pair may conflict with the longer gradient path of depth=7. Keep depth and pair variant separate.

---

## Exp 42 — Wider (node=128) + deeper (depth=7) + 5000 samples
- **Status:** Just submitted (job 18861545). Tests whether combining width and depth helps.

---

## Exp 43 — CosineAnnealingWarmRestarts instead of StepLR (5000 samples)
- **Result:** CANCELLED at ep0. val_force=152 vs exp 30 ep0 ~65-80. Cosine annealing from scratch is too unstable.
- **Takeaway:** Confirmed exp 12's finding — CosineAnnealingWarmRestarts with high LR from random init is unstable. StepLR is safer for training from scratch. Cosine restarts should only be tried from pre-converged checkpoints.

---

## Exp 44 — Larger equivariant channels (pos=32, vel=32) with 5000 training samples
- **Result:** CANCELLED at ep1. val_force=77 vs exp 30's 37.8 at ep1. Much worse than baseline despite 5000 samples + pair baseline.
- **Takeaway:** pos=32/vel=32 is still worse than pos=16/vel=16 even with 5000 samples. The overfitting is in the equivariant channels specifically. Stick with pos=16, vel=16 as the standard.

---

## General Lessons

1. **Angle features (3-body) help** — AngleToEdge consistently improves val_force.
2. **Residuals + SiLU** beat Tanh + no residuals from the base template.
3. **VelocityNormToNode** provides useful scalar geometric info and should always be included.
4. **model.eval() in validation is mandatory** — missing it corrupts val metrics.
5. **Full-epoch train averaging** gives more honest signal than last-batch.
6. **Save scheduler state** in checkpoint — otherwise LR restarts cause energy spikes.
7. **Pairwise baseline (learned position space) is the single biggest improvement** — halves val_force, eliminates overfitting.
8. **StepLR(10, 0.5)** is simpler and more resume-stable than cosine/warmup schedules.
9. **5000 training samples** gives 5× faster convergence per epoch (5× more gradient steps). Dramatically accelerates early training.
10. **Wider models (node=128) work with 5000 samples + pair baseline** — prevents overfitting that plagued exp 5 (96 features) and exp 20 (pos=32).
11. **Val sets differ by n_tr** — load_md17 places val at [n_tr, n_tr+n_vl). Experiments with different n_tr use different val sets. Comparisons are approximately valid but not exact.
12. **Dual pair (exp 32) has near-zero overfitting** but converges slower than standard pair. May not be better in the long run.
13. **Deeper models (depth=7+) need more data** — with 1000 samples, depth=8 was worse (exp 14). With 5000 samples, depth=7 may help (exp 37, 41, 42 testing).
14. **Never retry: vel norm² readout** (exps 18, 33), **Legendre P2/P3** (exp 21, 24), **edge energy readout** (exp 29), **batch-centered loss** (exp 9), **raw Cartesian pair** (exp 25).
15. **CosineAnnealingWarmRestarts from scratch is unstable** (exps 12, 43). Only try from pre-converged checkpoints.
16. **n_tr=10000 is NOT better than 5000** — at same gradient steps, exp 34 (10000) was dramatically worse than exp 30 (5000). Stick with 5000 as the primary data size.
17. **Exp 30** (5000 samples) reaches val_force=6.65 at epoch 8 — exp 22 needed 83 epochs for 6.0. 5000 samples is 10× more efficient in terms of epochs.
18. **Exp 36** (node=128 + 5000) at epoch 10 = 4.32 — NEW BEST. Near-zero overfitting. The LR decay at epoch 10 (3e-4 → 1.5e-4) triggers massive improvement jump.
19. **pos=32/vel=32 is worse even with 5000 samples** (exp 44). Equivariant channel size should stay at 16.
20. **depth=7 + DualPair combination is unstable** (exp 41). Combining multiple difficult-to-optimize elements creates convergence difficulties.
21. **Exp 39 (dual pair + node=128 + 5000)**: reaches 2.83 at epoch 14 — on track to break <1.0 by epoch 20-25. Best combination found.
22. **After LR decay**, dual pair outperforms standard pair dramatically. Dual pair + node=128 reaches 2.83 at ep14 vs standard pair + node=128 (exp 36) at 4.24 at ep12.
23. **Dual pair benefits from wider models**: dual pair + node=64 (exp 38) is much slower (ep8=15.30) than dual pair + node=128 (exp 39, ep8=6.94). The wider message-passing better utilizes both pair pathways.
