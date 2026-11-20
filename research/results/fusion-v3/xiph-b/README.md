# Protocol v3 on xiph-b: recoverability-guided fusion (RGF) and label-free adaptive calibration

Untouched test set; gates G5–G6 are judged here. Everything fitted came from the 20 dev clips. T* = 0.5 (chosen on dev by the pre-registered rule). The clip is the unit.

## G5: RGF vs RIFE 4.9 (all gaps and loss processes)

ΔPSNR = **+0.401 dB** [+0.346, +0.466], 8/8 clips better, Wilcoxon p = 0.0078. ΔLPIPS = +0.0342 [+0.0196, +0.0504].
**G5 FAILED**.

## All methods by gap length (PSNR dB / LPIPS, scene means)

| Gap | rife | amt_s_gopro | ifrnet_gopro | flow | linear | rgf |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 34.46 / 0.033 | 30.83 / 0.115 | 31.64 / 0.084 | 32.81 / 0.061 | 28.78 / 0.089 | **34.64 / 0.050** |
| 2 | 32.02 / 0.049 | 28.94 / 0.148 | 29.52 / 0.116 | 30.33 / 0.099 | 26.79 / 0.116 | **32.16 / 0.079** |
| 4 | 29.32 / 0.061 | 27.99 / 0.144 | 28.56 / 0.118 | 28.55 / 0.102 | 25.35 / 0.106 | **29.68 / 0.087** |
| 8 | 25.95 / 0.114 | 25.16 / 0.234 | 25.43 / 0.204 | 25.72 / 0.168 | 23.11 / 0.152 | **26.70 / 0.158** |
| 16 | 23.06 / 0.166 | 22.05 / 0.271 | 22.07 / 0.242 | 22.78 / 0.218 | 21.20 / 0.205 | **23.39 / 0.201** |
| 32 | 20.57 / 0.200 | 19.92 / 0.322 | 19.98 / 0.284 | 20.86 / 0.257 | 19.77 / 0.236 | **20.97 / 0.239** |
| bernoulli | 32.81 / 0.042 | 29.97 / 0.135 | 30.65 / 0.107 | 31.68 / 0.078 | 28.32 / 0.097 | 33.12 / 0.065 |
| gilbert_elliott | 28.79 / 0.083 | 27.39 / 0.173 | 27.66 / 0.151 | 28.09 / 0.124 | 25.17 / 0.133 | 29.18 / 0.110 |
| gop_chain | 27.11 / 0.087 | 25.90 / 0.189 | 26.18 / 0.158 | 26.51 / 0.137 | 23.96 / 0.134 | 27.56 / 0.119 |

**G5b PASSED**: worst per-gap margin of RGF over the best single candidate = +0.10 dB (floor −0.10).

## RGF vs each candidate (all tasks, paired over clips, Holm-corrected)

| vs | ΔPSNR | 95% CI | Clips better | Holm p |
|---|---:|---|---:|---:|
| rife | +0.40 | [+0.35, +0.47] | 8/8 | 0.0391 |
| amt_s_gopro | +1.64 | [+0.85, +2.59] | 8/8 | 0.0391 |
| ifrnet_gopro | +1.41 | [+0.81, +2.08] | 8/8 | 0.0391 |
| flow | +0.74 | [+0.05, +1.41] | 6/8 | 0.1094 |
| linear | +2.86 | [+1.32, +4.57] | 8/8 | 0.0391 |

## G6/G7: label-free adaptive calibration (RIFE errors, τ = 20, α = 0.1)

| Stratum | λ_dev | λ_target (no labels) | Deployed λ | Coverage (LFAC) | Area (LFAC) | Coverage (dev-only λ) | Area (dev-only λ) | Clips with bad pixels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gap1 / short | 0.019 | 0.053 | 0.019 | 0.966 | 0.132 | 0.966 | 0.132 | 5 |
| gap2 / short | 0.019 | 0.053 | 0.019 | 0.991 | 0.199 | 0.991 | 0.199 | 7 |
| gap4 / short | 0.019 | 0.053 | 0.019 | 0.991 | 0.298 | 0.991 | 0.298 | 7 |
| bernoulli / short | 0.019 | 0.053 | 0.019 | 0.944 | 0.161 | 0.944 | 0.161 | 8 |
| gilbert_elliott / short | 0.019 | 0.053 | 0.019 | 0.950 | 0.251 | 0.950 | 0.251 | 7 |
| gop_chain / short | 0.019 | 0.053 | 0.019 | 0.942 | 0.148 | 0.942 | 0.148 | 6 |
| gap8 / long | 0.037 | 0.151 | 0.037 | 0.988 | 0.473 | 0.988 | 0.473 | 8 |
| gap16 / long | 0.037 | 0.151 | 0.037 | 0.927 | 0.518 | 0.927 | 0.518 | 8 |
| gap32 / long | 0.037 | 0.151 | 0.037 | 0.933 | 0.584 | 0.933 | 0.584 | 8 |
| gilbert_elliott / long | 0.037 | 0.151 | 0.037 | 0.957 | 0.359 | 0.957 | 0.359 | 6 |
| gop_chain / long | 0.037 | 0.151 | 0.037 | 0.941 | 0.452 | 0.941 | 0.452 | 7 |

**G6 PASSED**: worst gated stratum coverage 0.927 (floor 0.88).
