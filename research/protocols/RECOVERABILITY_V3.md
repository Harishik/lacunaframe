# Protocol v3: recoverability-guided fusion and label-free adaptive calibration

Frozen 2026-11-20, **before** the Xiph-B set is downloaded, before the fusion method is fit, and before any v3 analysis. v1 and v2 remain the record of earlier claims. Machine-readable copy: [recoverability-v3.json](recoverability-v3.json).

## 1. Motivation (from v1/v2 evidence)

- No single reconstructor wins everywhere. RIFE 4.9 is perceptually best; the arbitrary-time AMT-S/IFRNet checkpoints tie it on PSNR; linear blending beats it at 32-frame gaps on SNU test. On dev, per-frame oracle selection gains only about 0.3 dB, but **per-pixel** oracle selection among {linear, flow, RIFE} lowers mean absolute error by 15%.
- Recoverability predicts error well and transfers across datasets (G2 ρ = 0.675, G2′ ρ = 0.881). That makes it a usable *gating signal* for per-pixel fusion.
- Dev-calibrated risk masks lost coverage under dataset shift (G3′ failed). The transferable recoverability model can estimate the target-domain error distribution *without target labels*.

## 2. Method A: recoverability-guided fusion (RGF)

- **Candidates (fixed):** RIFE 4.9, AMT-S (GoPro ×8), IFRNet (GoPro ×8), two-anchor flow, linear blending.
- **Per-pixel features (80×45 grid):** the 9 v1 recoverability channels, plus candidate-disagreement channels computed from the candidates' outputs only: each candidate's mean absolute deviation from the per-pixel candidate median, and the per-pixel standard deviation across candidates. Per-frame summaries (mean, 90th percentile) are appended. No reference frame is used.
- **Error predictors:** one gradient-boosted regressor per candidate, predicting that candidate's pooled absolute error. Same hyperparameters as v1; fit on the dev split (20 YouTube clips).
- **Fusion:** weights w_k = softmax(−ê_k / T) on the 80×45 grid, bilinearly upsampled to 640×360. The fused frame is Σ_k w_k · candidate_k. Observed frames are untouched.
- **Temperature T** ∈ {0.5, 1, 2, 4, 8}, chosen on dev only by 5-fold clip-grouped cross-validation (out-of-fold predictions) on a fixed subset of dev missing frames: every 4th missing frame of every task, with candidates stored temporarily and deleted afterwards. **Rule:** the highest dev PSNR among temperatures whose dev LPIPS ≤ RIFE 4.9's dev LPIPS + 0.005. If none qualifies, the lowest-LPIPS temperature.

## 3. Method B: label-free adaptive calibration (LFAC)

- **Score:** a gradient-boosted classifier, fit on dev, estimating p = P(RIFE pixel error > τ | features), with τ = 20 and the RGF features.
- **Dev λ:** conformal risk control on dev clips (α = 0.1), as in v1.
- **Target λ (no labels):** on the unlabelled target clips, the estimated false-negative rate of threshold λ is FNR̂(λ) = Σ_{p < λ} p / Σ p. λ_target is the largest λ with FNR̂ ≤ α.
- **Deployed threshold:** λ = min(λ_dev, λ_target), never less conservative than dev calibration. Mondrian strata (short ≤ 4 / long > 4) as in v1.

## 4. Untouched test set: Xiph-B

Ten 1080p 30 fps sequences from `media.xiph.org/video/derf/y4m/`, none used before: aspen, controlled_burn, red_kayak, snow_mnt, speed_bag, touchdown_pass, west_wind_easy (4:2:2), and dinner, factory, life (4:2:0). **rush_field_cuts is excluded in advance** because it contains editorial cuts.

Format: a centred 48-frame window at stride 1 (30 fps), fetched by HTTP range reads. 4:2:2 chroma is averaged vertically to 4:2:0, then converted with the same OpenCV I420 routine as Xiph v2, centre-cropped to 16:9 at 640×360. Tasks are identical to v1 (gaps 1–32, three loss processes at 25%).

## 5. Gates (judged on Xiph-B only)

| Gate | Pass condition |
|---|---|
| **G5** fusion beats the strongest single model | RGF − RIFE 4.9 affected-frame PSNR, paired over clips across all T3 tasks: 95% CI lower bound > 0, **and** mean LPIPS difference RGF − RIFE ≤ +0.005 |
| **G5b** fusion beats the best candidate per gap length | For each gap length, RGF PSNR ≥ the best single candidate's PSNR − 0.1 dB (no regression versus any fixed choice) |
| **G6** coverage under shift | LFAC coverage ≥ 0.88 in every Xiph-B stratum with ≥ 3 clips containing bad pixels |
| **G7** (descriptive) | LFAC flagged area vs dev-only calibration, and the dev-only coverage on Xiph-B for reference |

Secondary (not gated): RGF on the SNU test split and on Xiph v2, both of which were already seen.

## 6. Unchanged

The clip is the unit of analysis; bootstrap CIs, paired Wilcoxon with Holm, the PSNR cap of 60 dB, the 6 GB storage cap, and failure retention all carry over from v1/v2.

## Amendment 1 — 2026-11-20 (before any v3 evaluation; Xiph-B fetch still in progress)

**Finding.** The centred window of `aspen_1080p` contains two editorial cuts: consecutive-frame mean absolute differences of 58 and 70 against a within-shot median of about 5. Interpolating across a cut has no ground truth to recover, and only `rush_field_cuts` had been excluded in advance.

**Rule (mechanical, applied to every Xiph-B clip before any method is run on it).** A cut is a consecutive-frame mean absolute RGB difference greater than 25 **and** more than 4× the clip's median consecutive difference. Any clip whose 48-frame window contains a cut is **excluded**. The exclusion list and the per-clip difference statistics are written to `research/results/fusion-v3/xiph-b-cuts.json`. If fewer than 6 clips remain, G5/G6 are reported as underpowered rather than passed or failed.

## Results against the v3 gates (Xiph-B, 8 clips after the cut rule; 2026-11-21)

| Gate | Outcome |
|---|---|
| **G5** RGF beats RIFE 4.9 | **Failed** on the LPIPS clause. PSNR **+0.40 dB** [+0.35, +0.47], 8/8 clips (Wilcoxon p = 0.0078), but LPIPS +0.034 [+0.020, +0.050] against the +0.005 limit |
| **G5b** no regression vs the best single candidate | **Passed**: RGF is the best PSNR method at every gap length and loss process (worst margin +0.10 dB) |
| **G6** coverage under shift | **Passed**: worst gated stratum 0.927 ≥ 0.88 |
| **G7** (descriptive) | The label-free target threshold was larger than the dev threshold in both strata, so the deployed λ always equalled λ_dev. LFAC added no adaptation here, and G6's pass comes from the disagreement-augmented classifier scores calibrated on dev. Xiph-B also appears closer to dev than Xiph v2 was (RIFE 34.5 dB at g = 1) |

**Interpretation.** Per-pixel fusion guided by predicted error is a reliable *fidelity* improvement over every single model, on unseen data. Softmax blending, however, trades perceptual sharpness for PSNR (the perception–distortion trade-off). A method that wins on both needs reconstruction trained with a perceptual objective, not a post-hoc blend. That motivates training our own model next.
