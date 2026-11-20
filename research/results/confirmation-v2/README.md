# Protocol v2 confirmation on untouched Xiph 1080p (12 clips)

All models, conformal thresholds and τ′ were fit on the v1 dev split (20 YouTube clips) and frozen before Xiph was scored. Xiph differs in camera, content and frame rate (25 fps). Clips are the unit for CIs.

## G2′ recoverability transfer (RIFE)

Pooled per-frame Spearman ρ = **0.881** (pair-only 0.845). Per-clip Δρ vs pair-only = +0.086 [+0.021, +0.153], 7/12 clips better.
**G2′ PASSED.**

## G3′ calibration transfer (RIFE, τ = 20, dev-calibrated Mondrian λ)

| Stratum | Coverage | Flagged area | Clips with bad pixels |
|---|---:|---:|---:|
| gap1 / short (≤4) | 0.982 | 0.403 | 9 |
| gap2 / short (≤4) | 0.968 | 0.482 | 10 |
| gap4 / short (≤4) | 0.888 | 0.526 | 12 |
| bernoulli / short (≤4) | 0.956 | 0.435 | 11 |
| gilbert_elliott / short (≤4) | 0.977 | 0.404 | 9 |
| gop_chain / short (≤4) | 0.815 | 0.465 | 7 |
| gap8 / long (>4) | 0.835 | 0.530 | 12 |
| gap16 / long (>4) | 0.883 | 0.629 | 12 |
| gap32 / long (>4) | 0.966 | 0.792 | 12 |
| bernoulli / long (>4) | 0.924 | 0.128 | 1 |
| gilbert_elliott / long (>4) | 0.849 | 0.597 | 7 |
| gop_chain / long (>4) | 0.839 | 0.515 | 11 |

**G3′ FAILED**: worst gated stratum coverage 0.815 (floor 0.88).

## G4′ conditioning with dev-fixed τ′

τ′ = 18.14 (dev 80th percentile; 20% of dev pixels bad). On Xiph, 25.6% of pixels are bad.

| Score | Coverage | Flagged area |
|---|---:|---:|
| conditioned | 0.941 | 0.606 |
| pair-only | 0.951 | 0.630 |

Area ratio conditioned / pair-only = 0.961 (pass ≤ 0.85). **G4′ FAILED.**

## Descriptive: all methods on Xiph (contiguous gaps 1–32, scene means)

Plain AMT/IFRNet names are Vimeo90K-trained (midpoint) checkpoints; `*_gopro` are the official arbitrary-time checkpoints (no scene or camera overlap with Xiph).

| Method | PSNR (dB) | LPIPS | ΔPSNR vs RIFE 4.9 [95% CI] | Clips better | Holm p |
|---|---:|---:|---|---:|---:|
| hold | 18.60 | 0.181 | -2.80 [-4.18, -1.56] | 0/12 | 0.0044 |
| linear | 19.55 | 0.199 | -1.85 [-3.22, -0.62] | 4/12 | 0.1709 |
| flow | 20.90 | 0.211 | -0.49 [-1.12, -0.01] | 4/12 | 0.8145 |
| rife | 21.40 | 0.159 | — | — | — |
| amt_s | 19.48 | 0.284 | -1.92 [-3.13, -0.75] | 2/12 | 0.0732 |
| ifrnet_s | 19.31 | 0.253 | -2.09 [-3.34, -0.90] | 2/12 | 0.0547 |
| ifrnet | 19.32 | 0.244 | -2.08 [-3.32, -0.89] | 2/12 | 0.0547 |
| amt_s_gopro | 21.45 | 0.303 | +0.05 [-0.30, +0.35] | 7/12 | 1.0000 |
| ifrnet_s_gopro | 21.19 | 0.250 | -0.21 [-0.66, +0.18] | 5/12 | 1.0000 |
| ifrnet_gopro | 21.09 | 0.238 | -0.30 [-0.79, +0.12] | 5/12 | 1.0000 |
