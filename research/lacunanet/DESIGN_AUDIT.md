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
