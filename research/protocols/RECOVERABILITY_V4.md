# Protocol v4: LacunaNet, our own CPU-trained model for missing-run recovery

Frozen 2026-11-21, **before** any LacunaNet code, training, or v4 test download. Machine-readable copy: [recoverability-v4.json](recoverability-v4.json).

## 1. Goal and claim boundary

Build and train our own network, **entirely on a laptop CPU** (Intel Core i9-9880H, 8 cores, 16 GB; no GPU at any stage), that:

1. beats every reconstructor we have evaluated on LacunaBench missing runs, on fidelity **and** perceptual quality at once (the dual criterion that fusion failed in v3);
2. is no slower than RIFE 4.9 on the same CPU;
3. predicts its own per-pixel error, so recoverability and calibrated risk masks come from the same network.

On standard midpoint benchmarks (Vimeo90K, SNU-FILM) the claim is **Pareto efficiency** (quality vs CPU latency), not absolute superiority. The published baselines were trained on GPUs with the full training sets; that difference is disclosed, never hidden.

## 2. Prior-art position (2026-11-21 check)

More-than-two-frame interpolation exists but is rare and dated: QVI (NeurIPS 2019, 4 frames, quadratic motion), GDCN, Super SloMo-style arbitrary time. Self-predicted error exists for two-frame rendered-content interpolation (Plack et al., CVPR 2023). No CPU-native VFI benchmark exists; recent efficiency work targets NPUs (ANVIL) or GPUs. LacunaNet's novelty is the **combination**:

- multi-anchor, irregular-time context for contiguous missing runs, conditioned on gap length;
- a jointly trained heteroscedastic self-error head whose output is calibrated with conformal risk control;
- CPU-only training and inference, with a CPU latency constraint built into model selection.

## 3. Architecture (fixed family)

- **Inputs:** inner anchors A₋₁, A₊₁ (always present) and outer anchors A₋₂, A₊₂ (optional, with availability masks; zeros when absent). Also the normalised relative time of every anchor, the target position t ∈ (0, 1), and log₂ gap length.
- **Encoder:** a shared three-level pyramid of depthwise-separable convolution blocks (widths C, 2C, 4C), applied to every anchor.
- **Motion:** coarse-to-fine estimation (3 levels) of flows from the target to each inner anchor. Outer-anchor features, backward-warped by the current estimate, enter every level as motion-trajectory context. All levels are FiLM-conditioned on (t, gap, anchor times, masks).
- **Synthesis:** backward-warp the inner anchors, blend with a predicted occlusion mask, and refine with a small U-Net residual.
- **Self-error head:** predicts a per-pixel Laplace scale b(x) from the decoder features.
- **Width C** is chosen by the latency rule in §5. Nothing else is tuned on test data.

## 4. Training (CPU-only, fixed before training)

- **Data:**
  - (a) GOPRO_Large **train** sharp sequences (22 scenes, 2,103 frames, range-extracted from the official zip mirror), downscaled to 640×360. These give multi-anchor samples with gaps g ∈ {1, …, 16} and random target positions; outer anchors are present with probability 0.8.
  - (b) Vimeo90K triplet **train** mirror (25,656 triplets, about 50% of the official training set), streamed shard by shard and deleted after use. These give g = 1 midpoint samples with outer anchors absent.
  - Random 256×256 crops, flips and temporal reversal.
- **Disclosure:** GOPRO training scenes are disjoint from the GOPRO test scenes behind SNU-FILM's GoPro clips, but share the camera. Vimeo90K training is disjoint from its official test set.
- **Loss:**
  - Charbonnier reconstruction, plus 0.1 × LPIPS-AlexNet perceptual loss, plus the Laplace NLL for the self-error head.
  - In the NLL, the reconstruction residual is stop-gradiented into b, so the error head cannot degrade reconstruction.
- **Optimiser:**
  - AdamW, lr 3e-4 with cosine decay, batch 8, weight decay 1e-4.
  - **Budget: 48 wall-clock hours of CPU training**, with checkpoints every 2 hours.
- **Checkpoint selection:** on dev only (SNU YouTube clips, LacunaBench T3, scene-mean PSNR − 20 × LPIPS). The test sets are never consulted.
- **Data mixture:** 70% GOPRO, 30% Vimeo samples.

## 5. Model-size rule (CPU-native constraint)

Measure single-frame CPU latency at 640×360 (2 intra-op threads, ONNX Runtime, warm, median of 20) for C ∈ {16, 24, 32, 48}. Choose the **largest C whose latency ≤ RIFE 4.9's latency** under identical conditions. That makes "no slower than RIFE" true by construction.

## 6. Evaluation sets

| Set | Use |
|---|---|
| **Xiph-C (untouched, primary):** 720p50 mobcal, parkrun, shields; 720p59.94 stockholm; vidyo1, vidyo3, vidyo4 (720p60); plus CGI elephants_dream and sintel_trailer (1080p24) | Centred 48-frame windows, frame rate normalised by stride to ≤ 30 fps (stride 2 for 50/60 fps), the v3 scene-cut rule, minimum 6 clips. Gates G8, G11 |
| **Vimeo90K, full official test** (3,782 triplets, streamed) | Midpoint benchmark, per-image RGB PSNR/SSIM + LPIPS. G10 |
| **SNU-FILM, full official lists** (easy, medium, hard, extreme; 310 triplets each, 1280×720, streamed) | Midpoint benchmark at native resolution. G10 |
| SNU test, Xiph, Xiph-B (already seen) | Secondary, descriptive |

Every method, LacunaNet and all baselines, runs on the same CPU with the same thread budget, the same inputs and the same metric code. There is no subsampling of official benchmarks.

## 7. Gates

| Gate | Pass condition |
|---|---|
| **G8** dual win on missing runs (Xiph-C) | LacunaNet − RIFE 4.9: PSNR paired-over-clips 95% CI lower bound > 0 **and** mean LPIPS difference ≤ 0 (LacunaNet at least as sharp). Reported against every baseline and against RGF fusion |
| **G9** CPU efficiency | LacunaNet's CPU latency ≤ RIFE 4.9's at 640×360 (enforced by §5, re-measured at evaluation); parameters and file size reported |
| **G10** standard benchmarks (Pareto) | On Vimeo90K test and each SNU-FILM level, no evaluated baseline is both faster on our CPU **and** higher in PSNR than LacunaNet |
| **G11** built-in recoverability and calibration (Xiph-C) | The self-error head's per-frame Spearman ρ with LacunaNet's actual PSNR ≥ 0.6, **and** dev-calibrated conformal masks (clips as units, α = 0.1, Mondrian short/long) reach coverage ≥ 0.88 in every stratum with ≥ 3 clips containing bad pixels |

Failures are reported. If G8 fails, the result is described as such; nothing is retrained against test data.

## Amendment 1 — 2026-11-21 (before any training; no data used)

**Finding.** With the first implementation, the §5 size rule selected C = 16 (0.32M parameters; RIFE 4.9 = 0.54 s/frame, C = 24 = 0.74 s). Profiling C = 32 showed 73% of CPU time in dense 3×3 convolutions at full and half resolution (full-resolution refinement ≈ 25 GMAC). The encoders cost only 0.12 s.

**Change (compute placement only; same family, same inputs and outputs).**
1. Refinement runs at 1/2 resolution. Full-resolution evidence (blend, warped anchors, mask) enters by pixel-unshuffle, and the residual and error scale leave through pixel-shuffle heads.
2. Motion-level input projections are 1×1 instead of 3×3; spatial context comes from the depthwise-separable blocks.

**Re-applied rule (identical procedure):** latencies 0.18 / 0.32 / 0.42 / 0.74 s for C = 16 / 24 / 32 / 48 against RIFE 4.9 at 0.53 s, so **C = 32 is selected** (462751 parameters, about 22% faster than RIFE 4.9 on this CPU). Record: `research/results/lacunanet/size-rule.json`.
