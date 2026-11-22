# LacunaNet v1 design audit and v2 redesign

2026-11-21. Training of v1 was stopped at step 4,300 to save GPU quota. The diagnosis below uses checkpoint 2,500 on 6 dev clips (558 missing frames, all T3 tasks). Dev only; no test data was used.

## 1. Evidence: v1 does not learn motion

| Stratum | Linear | Two-anchor flow | RIFE 4.9 | **v1 @2.5k** |
|---|---:|---:|---:|---:|
| gap 1 | 28.45 | 32.26 | 33.84 | **28.39** |
| gap 2 | 26.23 | 28.46 | 30.23 | **26.31** |
| gap 3–4 | 25.15 | 26.98 | 28.21 | **25.07** |
| gap 9–16 | 21.32 | 21.84 | 23.45 | **21.60** |
| displacement 0–2 px | 25.67 | 27.65 | 30.35 | **25.57** |
| displacement 10–20 px | 18.11 | 18.06 | 18.64 | **18.66** |

v1 tracks linear blending to within ±0.3 dB everywhere. It is worst exactly where motion is easiest (0–2 px), so the failure is in motion estimation, not in difficulty. Raw record: `research/results/lacunanet/diagnosis-ckpt-02500.json`.

## 2. Root causes (architecture and mathematics)

1. **No explicit matching.** v1's motion levels regress flow from *concatenated* anchor features, with no correlation or cost volume. A network can only infer displacement from concatenation if its receptive field spans that displacement *and* it learns an implicit matcher. That is slow to learn, and it fails for long gaps. Every competitive interpolator either computes matching costs (PWC, IFRNet, AMT) or uses very wide coarse blocks on raw images (RIFE).
2. **The flow gradient has no capture range.** Flows start at zero and receive gradients only through bilinear `grid_sample`. That gradient is non-zero only within about one pixel of the current estimate. For displacements beyond about 1 px at a given level, the reconstruction loss gives almost no signal toward the correct flow. With zero-initialised flow heads, the easiest descent direction is the refinement residual, which is exactly what we observe.
3. **Wrong motion parameterisation for missing runs.** v1 predicts target→anchor flows *directly as a function of t*. It is therefore forced to learn the time dependence of motion, which physics already gives us. The anchor-pair motion is time-independent and can be estimated once per gap. The intermediate motion then follows from a trajectory model.
4. **Outer anchors are effectively unused.** v1 warps outer-anchor features with a *linear extrapolation* of the inner flow: ratio τ_outer/τ_inner, which is constant velocity by construction. The only information outer anchors carry beyond the inner pair is acceleration, and that warp assumes it is zero. Outer anchors enter only as weak context features.
5. **No motion supervision at all.** Only the target frame supervises motion, through the warping bottleneck above. The anchors themselves provide a free, dense, time-independent motion signal: the inner pair must warp onto each other. v1 ignores it.
6. **Training mix.** 30% of samples are Vimeo triplets (small motion, g = 1, no outer anchors). That is fine as data, but it gives no pressure to learn large motion while the motion branch is still dead.

## 3. v2 design (each change answers a cause above)

**Notation.** Inner anchors A₀ (time 0) and A₁ (time 1, in inner-span units), target time t ∈ (0, 1), optional outer anchors A₋ at −s₋ and A₊ at 1 + s₊.

1. **Matching pyramid (causes 1, 2).** A shared depthwise-separable encoder at 1/2, 1/4, 1/8 and 1/16. Coarse-to-fine **pair flow** F₀→₁ (and F₁→₀) is estimated PWC-style: at each level, backward-warp the other anchor's features with the current flow, compute a **local correlation volume** (radius 3, 49 channels, normalised dot product), and regress a flow update from [features, correlation, current flow]. The 1/16 level with radius 3 covers ±48 px at full resolution, enough for the long-gap displacements measured above. Correlation is computed at ≤ 1/4 resolution only (CPU budget).
2. **Physics-based trajectory (causes 3, 4).** With outer pair flows F₀→₋ and F₁→₊ (same estimator, shared weights, coarse levels only), fit per pixel a constant-acceleration trajectory from each inner anchor:
   - d₀(τ) = v₀τ + ½a₀τ², with a₀ = 2(F₀→₁·s₋ + F₀→₋·1) / (s₋·1·(s₋+1)) and v₀ = F₀→₁ − ½a₀ (the same closed form as `backend.core.quadratic_displacement`).
   - When an outer anchor is absent, a = 0 (constant velocity). Symmetrically, d₁ runs from A₁ toward the past.
3. **Differentiable flow reversal.** The forward displacements d₀(t) and d₁(t−1) live on anchor grids. Backward target→anchor flows B₀ and B₁ solve p + d(p) = q by K = 4 fixed-point iterations of `grid_sample` at 1/4 resolution. This is exactly invertible for smooth motion and differentiable end to end.
4. **Learned correction where physics stops.** A small FiLM-conditioned network at 1/4 resolution takes the warped anchor features, B₀, B₁, the forward/backward pair-flow inconsistency (an occlusion cue) and the conditioning vector. It predicts residual flow updates and the occlusion/blend logit. Upsampling, blending, the half-resolution refinement and the self-error head are kept from v1.
5. **Free motion supervision (cause 5).** An added anchor loss: Charbonnier(warp(A₁, F₀→₁), A₀) + Charbonnier(warp(A₀, F₁→₀), A₁), down-weighted where forward/backward flows disagree (occlusion), plus an edge-aware smoothness term. Every training sample supervises motion densely, independent of t.
6. **Initialisation sanity.** With zero pair flows, v2 still reduces exactly to temporal linear blending (unit-tested), so the comparison with v1 starts from the same point.

## 4. Verification before any GPU time

- Unit tests: linear-blend identity at init; the trajectory equals the analytic quadratic; flow reversal inverts a known smooth field; no leakage from absent outer anchors; ONNX parity.
- Size rule re-applied unchanged (≤ RIFE 4.9 latency on this CPU).
- **CPU learning-signal check:** v1 and v2, equal tiny budget, same data and seed. v2 must beat linear blending on held-out dev frames where v1 does not. If it doesn't, the redesign is wrong and is fixed before Colab.

## 5. Second round: v2's first CPU check failed, and why (2026-11-21)

The first equal-budget CPU check (400 steps, batch 4) had v2 *below* linear blending and below v1 (gap 1: 24.18 vs 28.42 dB). Each problem below was found by a targeted test and fixed. None was fixed by tuning numbers.

| # | Finding | Test that exposed it | Fix |
|---|---|---|---|
| 1 | **Bug:** on the A₁ side the optional neighbour is the *next* anchor, but the trajectory helper treated its optional neighbour as the *previous* one. Without Aₚ (every Vimeo sample, about 20% of GOPRO) the A₁ motion collapsed to zero | Regression test with a known constant pair flow | A₁ uses constant velocity f₁₀·(1−t) when Aₚ is absent, and the quadratic otherwise |
| 2 | The motion loss acted only at the finest level, outside the bilinear capture range of large shifts (the same failure diagnosed in v1) | Pair-flow-only learning test: no progress | Anchor loss at every pyramid level on area-downsampled anchors (1/16, 1/8, 1/4) |
| 3 | The coarsest flow started at zero | same | Coarse flow initialised by a **soft-argmax over the cost volume**, an explicit differentiable match |
| 4 | Zero padding in the cost volume biased border matches inward, creating false motion at frame edges | Static-scene test (identical anchors must give zero motion) | Replicate padding, plus out-of-image offsets excluded from the soft-argmax |
| 5 | **Cosine similarity on post-activation (mostly positive) features** is high at every offset (0.91 wrong vs 1.00 true), so the soft-argmax could not separate the true match | Known-shift test: the argmax channel was right, the soft-argmax was not | **Pearson** correlation (centre, then normalise): a (2, −1) shift is now recovered as (1.98, −0.99) |
| 6 | Aperture ambiguity: on low-texture regions many offsets score alike, and the expectation drifts | Static-scene test at corners | Weak zero-motion prior in the soft-argmax logits (0.5 per px² at 1/16) and a sharper initial temperature (0.02). A distinctive match overrides the prior |
| 7 | My own synthetic test was invalid: blurred uniform noise has no coarse-scale structure to match | Same test on real GOPRO frames | Real-frame known-shift test: flow error falls from 4.5 to 2.6 px in 400 steps (zero-guess baseline 6.5 px) and is still decreasing |

All v2 unit tests pass after these fixes (8 tests). The equal-budget v1 vs v2 comparison is re-run with the fixed model before any GPU time.

## 6. Third round: the bottleneck is learning the pair flows, so the motion gets a classical prior (v3)

The fixed v2 still did not beat linear blending at equal budget (400 steps: gap 1 = 27.37 vs 28.42 dB). Rather than tune, the motion path was taken apart.

**Oracle test** (`oracle_pipeline.py`, 6 dev clips, no training). Classical pair flows go through v2's *own* trajectory → reversal → warp → blend path at quarter resolution, with the learned correction and refinement off:

| PSNR (dB) | gap 1 | gap 2 | gap 4 | gap 8 |
|---|---:|---:|---:|---:|
| Linear blending | 29.14 | 26.82 | 23.52 | 23.07 |
| v2 pipeline + Farnebäck flows (OpenCV) | 33.95 | 29.90 | 26.20 | 25.70 |
| v2 pipeline + differentiable Lucas–Kanade, full-res pyramid | 32.55 | 29.35 | 26.46 | 25.17 |
| v2 pipeline + differentiable Lucas–Kanade at 1/4 resolution | 31.88 | 29.09 | 26.22 | 25.15 |

So the physics path (trajectory, fixed-point reversal, quarter-resolution warping) is sound: with good flows it gains 2.1–4.8 dB over linear blending. **The whole deficit is in learning the pair flows.** A matcher learned from scratch with a photometric loss starts out worse than predicting zero motion (end-point error 5.4 px vs 4.4 px for zero flow at gap 1, measured against Farnebäck). It only becomes useful after far more updates than any small equal-budget check allows, and until then warping by wrong flows is worse than not warping.

Two attempts that did **not** fix it, recorded for completeness:
- A parameter-free NCC cost volume (contrast-normalised 3×3 patches) as a second matching volume. Coarse-to-fine soft-argmax over it was no better than zero flow: patch NCC is ambiguous on low-texture video, and errors at 1/16 are amplified 16× on the way up.
- End-point error against Farnebäck as the yardstick. Lucas–Kanade recovers synthetic shifts to within 0.02 px (Farnebäck: within 10%), yet scored "worse than zero flow" against Farnebäck on real pairs. Farnebäck is not ground truth on real motion. Reconstruction PSNR through the pipeline is the measure that matters, and it is the one used above.

**v3 design: classical motion prior + learned residual** (`classical_flow.py`, `PairFlow(prior='lk')`).
- **Prior.** A pyramidal Lucas–Kanade solver (Gauss–Newton on the brightness-constancy linearisation, Gaussian-weighted 2×2 structure tensor with Tikhonov damping λ, warping between iterations). It is unrolled inside the network from convolution, pooling and `grid_sample` only, so it is differentiable, has no parameters and exports to ONNX. Inner pairs are solved at 1/4 resolution and outer pairs at 1/8.
- **Learned residual.** At every pyramid level (1/16, 1/8, 1/4) the flow is the prior resampled to that level plus a learned residual carried coarse to fine. Each head reads the cost volume *around the prior* (anchor B warped by prior + residual). Heads are zero-initialised, so at step 0 v3's motion *is* the Lucas–Kanade estimate. Learning only has to correct where the classical model fails: large displacements outside its linearisation range, occlusions, and aperture problems. The multiscale anchor loss supervises prior + residual at every level.
- **Everything downstream is unchanged:** trajectory, reversal, correction network, refinement and self-error head. So v2 vs v3 isolates the motion prior.
- Tests: an untrained v3 recovers a (6, −3) px shift of real texture to within 0.3 px, and identical anchors reproduce the input exactly.

## 7. Fourth round: why training made every model worse than its own starting point

In the equal-budget check (1,000 steps, `cpu-sanity-v1-v2-v3-1000.json`) v3 beat linear blending and v1 at every gap, but training *lowered* it below its untrained self (gap 1: 31.87 → 30.81 dB). v1 fell 3.5 dB below linear blending, its own initial state. The training loss bounced between 0.05 and 0.14 with no trend. Three root causes were found. None is a hyperparameter.

**7.1 Training data from the wrong motion regime** (`motion_stats.py`, `motion-stats.json`). Inner-anchor motion, measured identically (Lucas–Kanade, full-frame mean, px):

| Motion bin (px) | 0–2 | 2–4 | 4–8 | 8–16 | 16–32 | 32–64 | ≥64 | median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dev missing-frame brackets | 0.09 | 0.11 | 0.18 | 0.29 | 0.18 | 0.14 | 0.03 | 11.7 |
| GOPRO training samples as drawn | 0 | 0 | 0.03 | 0.05 | 0.17 | 0.44 | 0.31 | 49.1 |

GOPRO_Large is handheld *deblurring* footage, chosen for strong camera shake. Its sharp frames are about 1/30 s apart, so gaps up to 16 span up to half a second of shaking. 75% of training samples moved more than 32 px, and none moved less than 4 px. On largely unrecoverable samples, the loss-minimising prediction is a hedge between the possible outcomes, which is blur. The networks learned exactly that, and it degraded the easy dev cases.

**Fix: `StratifiedMotion` sampler** (`data.py`). Every GOPRO anchor pair (gap ≤ 16; 20,168 pairs) and every Vimeo-90K *train* triplet gets its exact motion once, by batched Lucas–Kanade. A sample is then drawn by picking a motion bin from the target histogram, then a pair that can land in it. GOPRO pairs are resized by s ∈ [0.72, 1] chosen to put s·m inside the bin, which is exact because displacement scales with resolution. Vimeo supplies most of the small-motion bins. The target is the dev bracket histogram: statistics only, no dev frames are trained on. Measured on the produced crops, the result is 0.085 / 0.15 / 0.17 / 0.24 / 0.18 / 0.09 / 0.09 against the target 0.088 / 0.11 / 0.18 / 0.29 / 0.18 / 0.14 / 0.03, at 9 ms per sample. A first rejection-sampling version accepted only 8% of draws, because the proposal hardly overlaps the target, so it was replaced. Limitation: only 25 source pairs move less than 2 px, so that bin repeats content (crops and flips vary it). Every new Vimeo shard adds more.

**7.2 Unaligned context features in the refinement (v1 and v2/v3).** The refinement received `0.5·e0 + 0.5·e1`: the two anchors' encoder features averaged *without warping to the target time*. For moving content that is a ghosted, linear-blend-like signal, which tells the refinement network the answer is the blend. This matches the measured symptom: refinement output below its own warped base (30.81 vs 31.57 dB at gap 1). **Fix:** context features are warped with the same backward flows and occlusion mask as the pixels. v1 (`model.py:147`) has the same defect. It is left as the frozen v1 baseline and recorded here.

**7.3 Uncertainty head reshaping the reconstruction.** The self-error head shared its trunk with the reconstruction head, so its likelihood loss sent gradients into the features that produce the frame. **Fix:** the error head reads the trunk through a stop-gradient. The uncertainty estimate now describes the reconstruction without changing it, which is also what the conformal-calibration claim (C3) assumes.

**Also fixed for GPU training:** the Lucas–Kanade solver now always runs in float32. Under mixed-precision autocast, gradient products (~1e-4) and the 2×2 determinant underflow in float16. The Colab trainer now uses the shared samplers in `data.py` instead of its own duplicate copies, which is how the sampling bug could have come back.

**Result of the section 7 fixes** (1,000 steps, 6 dev clips; `cpu-sanity-v3-1000-matched.json`, `cpu-sanity-v3-1000.json`):

| v3, PSNR (dB) | gap 1 | gap 2 | gap 4 | gap 8 |
|---|---:|---:|---:|---:|
| Untrained (Lucas–Kanade prior) | 31.87 | 29.41 | 26.96 | 25.60 |
| Before fixes, trained | 30.81 | 29.05 | 26.09 | 25.31 |
| Fixed, GOPRO-only data | 31.00 | 29.12 | 26.01 | 25.23 |
| Fixed, stratified data | 31.52 | 29.17 | 26.19 | 24.98 |
| Linear blending | 29.14 | 26.82 | 23.52 | 23.07 |

The refinement penalty shrank (gap 1: output minus warped base −0.76 → −0.27 dB) and the training loss fell from about 0.09 to about 0.03. But the warped base itself still drifts below the prior (gap 4: 26.96 → 26.33). The learned motion corrections damage a good estimate early in training.

## 8. Fifth round: why the learned corrections damage the prior

**8.1 Conflicting per-level supervision.** The residual is carried coarse to fine (1/16 → 1/8 → 1/4), and v2's multiscale anchor loss gave *each level's own flow* its own photometric target, weighted coarsest-first (1.0 / 0.5 / 0.25). That was right for v2, which had to bootstrap motion from nothing. With a fine-level prior it is backwards. The 1/16 photometric optimum differs from the fine truth because pooling mixes motions, so the strongest signal trains the coarse residual toward a blurry optimum, and that correction is then added onto the precise fine prior. **Fix (`anchor_loss_final`):** only the final 1/4 flow is supervised. It is average-pooled to 1/8 and 1/16 for the coarse photometric terms (keeping capture range), with full weight at the finest scale, which is the one actually used.

**8.2 Zero-initialised residual outputs under Adam.** Adam's first updates move every weight by about the learning rate, roughly by gradient sign. Zero-initialised output convolutions therefore start emitting small unstructured offsets onto an accurate prior before they learn anything. **Fix: ReZero gates.** Each residual output (flow heads, flow/blend correction, image refinement) is normally initialised and multiplied by a learnable scalar that starts at 0. The gate's gradient pools evidence over every pixel, so a correction enters only as it lowers the loss on average. Output at step 0 is still exactly the prior (unit-tested).

**8.3 ONNX export of the classical layer.** The Gaussian kernel was built with tensor operations, and the prior was resampled by area interpolation with a runtime size. Neither exports. The kernel is now built from Python constants, and the prior is pooled by the structurally fixed power-of-two ratios between pyramid levels.

**Result of the section 8 fixes** (`cpu-sanity-v3-1000-matched.json`, 1,000 steps): gap 1/2/4/8 = 31.62 / 29.33 / 26.74 / 25.42 dB. That is within 0.1–0.25 dB of the untrained prior (previously 0.25–0.8 dB below). The refinement is now neutral: output and warped base agree to within 0.01 dB.

## 9. Sixth round: the anchor photometric loss is a misaligned objective for v3

**Component ablation** (`ablate.py`, `ablation-gated-v3.json`): each learned part of the trained model is switched off by zeroing its gate.

| Variant, dB | gap 1 | gap 2 | gap 4 | gap 8 |
|---|---:|---:|---:|---:|
| Trained | 31.62 | 29.33 | 26.74 | 25.42 |
| Without pair-flow residual | 31.78 | 29.34 | 26.91 | 25.55 |
| Without flow correction | 31.64 | 29.36 | 26.79 | 25.46 |
| Without blend correction | 31.60 | 29.33 | 26.75 | 25.47 |
| Without refinement | 31.63 | 29.37 | 26.74 | 25.42 |
| Prior only | 31.87 | 29.41 | 26.96 | 25.60 |

Almost all of the remaining loss comes from the learned pair-flow residual. Its finest gate learned the largest value (−0.117).

**Misalignment test.** On all 24 dev anchor pairs, the learned residual *lowers* the anchor photometric loss (0.0670 → 0.0618) while *lowering* reconstruction PSNR. The anchor loss rewards flows that match brightness, not flows that track real motion: under occlusion, low texture or lighting change the photometric optimum is not the true motion. Lucas–Kanade is already the regularised solution of that same objective. The loss was introduced (section 3.5) to give v2's from-scratch flows a dense bootstrap signal. v3 gets its capture range from the prior, so for v3 the anchor loss is a misaligned proxy.

**Attempted fix, then reverted on stronger evidence.** I retrained v3 with its motion trained only through the target reconstruction (r9). On the 6-clip check it looked better at gap 1. On **all 20 dev clips** (`paired_eval.py`, `paired-eval-v3.json`: per-frame paired differences, 95% CI by bootstrap over clips), it is significantly *worse* than keeping the final-flow anchor loss (r8):

| Paired difference (dB), 95% CI | gap 1 | gap 2 | gap 4 | gap 8 | all (300 frames) |
|---|---|---|---|---|---|
| r8 (final-flow anchor loss) − prior | −0.08 [−0.36, 0.10] | −0.09 [−0.18, −0.00] | −0.14 [−0.28, −0.03] | −0.08 [−0.18, 0.00] | −0.10 [−0.20, −0.02] |
| r9 (reconstruction only) − prior | −0.02 [−0.09, 0.04] | −0.13 [−0.20, −0.07] | −0.20 [−0.38, −0.05] | −0.19 [−0.37, −0.06] | −0.17 [−0.32, −0.07] |
| r9 − r8 | 0.06 [−0.10, 0.30] | −0.04 [−0.08, 0.01] | −0.06 [−0.11, −0.02] | −0.11 [−0.21, −0.03] | −0.07 [−0.13, −0.03] |
| Linear blending − prior | −3.42 | −2.99 | −2.90 | −1.88 | −2.40 [−3.62, −1.36] |

The anchor loss is a biased objective (the misalignment measurement stands), but it still regularises the residual, especially at large gaps. It is restored for v3. Two lessons:
1. **The 6-clip check was underpowered.** Gap 1 had 6 frames. Every CPU check now evaluates all 20 dev clips with paired clip-clustered CIs.
2. **At this budget the learned layers cannot beat the prior.** After all the fixes, trained v3 sits 0.10 dB [0.02, 0.20] below the prior on full dev, down from about 1 dB below before sections 7–8. 1,000 steps at batch 4 is 4,000 samples, about 0.4% of the planned GPU budget (60,000 × 16). The learned corrections have barely started and their gates are near zero. Whether learning helps can only be answered with the real budget. Do-no-harm is ensured by checkpoint selection on dev **including the step-0 prior-only model as a candidate**, so the delivered model cannot be worse on dev than the classical pipeline, which is itself +2.40 dB [1.36, 3.62] over linear blending.

**Size rule for v3, idle CPU** (`size-rule-v3-idle.json`; interleaved timing, RIFE 4.9 at 0.554 s, matching the original 0.563 s idle measurement): latency ratio to RIFE for C = 16 / 24 / 32 / 48 is 0.56 / **0.93** / 1.23 / 2.13. **C = 24 is kept.** Under the other project's load the same models measured 1.13× (v3) and 1.08× (v2): LacunaNet's many small operations suffer more from CPU contention than RIFE's fewer large ones. Only idle, interleaved measurements are valid for the rule.
