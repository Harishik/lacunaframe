# LacunaBench v1: results (task T3, SNU-FILM)

The scene (clip) is the unit: frames are averaged within each clip, and 95% CIs bootstrap over clips. Dev = 20 YouTube clips, test = 11 GoPro clips (frozen in `research/protocols/splits-v1.json`). PSNR is capped at 60 dB before averaging. DISTS is scheduled per amendment 1.

## dev: affected-frame PSNR (dB) by gap length

| Gap | hold | linear | flow | flow_quadratic | rife |
|---|---:|---:|---:|---:|---:|
| 1 | 23.56 [21.15, 26.18] | 26.42 [23.71, 29.30] | 29.98 [26.58, 33.55] | 30.27 [26.82, 33.87] | 31.93 [28.29, 35.64] |
| 2 | 23.72 [21.30, 26.21] | 25.13 [22.62, 27.75] | 27.28 [24.75, 29.89] | 27.83 [25.17, 30.57] | 29.17 [26.53, 31.99] |
| 4 | 21.33 [19.30, 23.48] | 22.49 [20.40, 24.71] | 24.04 [21.36, 26.99] | 24.67 [21.81, 27.84] | 25.64 [22.93, 28.56] |
| 8 | 20.68 [18.73, 22.70] | 21.67 [19.70, 23.71] | 22.24 [20.01, 24.65] | 22.72 [20.43, 25.15] | 23.59 [21.03, 26.37] |
| 16 | 19.29 [17.59, 21.01] | 20.19 [18.50, 21.92] | 20.38 [18.55, 22.28] | 20.48 [18.60, 22.41] | 21.25 [19.22, 23.41] |
| 32 | 17.67 [16.08, 19.32] | 18.54 [16.95, 20.18] | 18.41 [16.78, 20.11] | 18.26 [16.60, 20.03] | 19.01 [17.21, 20.84] |

## dev: LPIPS (lower is better) by gap length

| Gap | hold | linear | flow | flow_quadratic | rife |
|---|---:|---:|---:|---:|---:|
| 1 | 0.097 | 0.110 | 0.090 | 0.091 | 0.054 |
| 2 | 0.095 | 0.112 | 0.110 | 0.117 | 0.059 |
| 4 | 0.135 | 0.155 | 0.167 | 0.174 | 0.093 |
| 8 | 0.164 | 0.183 | 0.207 | 0.218 | 0.130 |
| 16 | 0.215 | 0.233 | 0.272 | 0.295 | 0.182 |
| 32 | 0.277 | 0.291 | 0.335 | 0.377 | 0.254 |

## dev: loss processes at 25% loss (PSNR dB / LPIPS)

| Process | hold | linear | flow | flow_quadratic | rife |
|---|---:|---:|---:|---:|---:|
| bernoulli | 23.13 / 0.102 | 25.28 / 0.118 | 28.36 / 0.109 | 28.64 / 0.112 | 30.21 / 0.060 |
| gilbert_elliott | 22.45 / 0.130 | 23.96 / 0.145 | 25.92 / 0.156 | 26.32 / 0.162 | 27.32 / 0.094 |
| gop_chain | 21.56 / 0.147 | 22.73 / 0.168 | 23.82 / 0.182 | 24.24 / 0.190 | 25.44 / 0.104 |

## test: affected-frame PSNR (dB) by gap length

| Gap | hold | linear | flow | flow_quadratic | rife |
|---|---:|---:|---:|---:|---:|
| 1 | 18.03 [16.51, 19.64] | 19.98 [18.42, 21.50] | 22.91 [20.19, 25.41] | 23.25 [21.10, 25.34] | 23.70 [20.50, 26.66] |
| 2 | 17.86 [16.68, 19.13] | 19.03 [17.80, 20.42] | 21.35 [19.64, 23.19] | 21.85 [20.48, 23.39] | 22.21 [20.11, 24.41] |
| 4 | 16.67 [15.46, 17.99] | 17.70 [16.57, 18.95] | 18.73 [17.21, 20.40] | 19.11 [17.59, 20.78] | 19.55 [17.80, 21.44] |
| 8 | 16.19 [15.05, 17.48] | 17.07 [15.95, 18.34] | 17.29 [16.14, 18.57] | 17.54 [16.36, 18.85] | 17.80 [16.26, 19.46] |
| 16 | 14.65 [13.83, 15.46] | 15.50 [14.68, 16.31] | 15.44 [14.61, 16.27] | 15.30 [14.43, 16.17] | 15.76 [14.77, 16.79] |
| 32 | 13.75 [12.95, 14.67] | 14.57 [13.75, 15.50] | 14.36 [13.54, 15.30] | 14.06 [13.27, 14.94] | 14.45 [13.58, 15.39] |

## test: LPIPS (lower is better) by gap length

| Gap | hold | linear | flow | flow_quadratic | rife |
|---|---:|---:|---:|---:|---:|
| 1 | 0.147 | 0.184 | 0.122 | 0.161 | 0.084 |
| 2 | 0.138 | 0.174 | 0.134 | 0.163 | 0.099 |
| 4 | 0.192 | 0.217 | 0.199 | 0.242 | 0.137 |
| 8 | 0.231 | 0.251 | 0.257 | 0.287 | 0.191 |
| 16 | 0.311 | 0.326 | 0.351 | 0.385 | 0.264 |
| 32 | 0.386 | 0.392 | 0.423 | 0.470 | 0.361 |

## test: loss processes at 25% loss (PSNR dB / LPIPS)

| Process | hold | linear | flow | flow_quadratic | rife |
|---|---:|---:|---:|---:|---:|
| bernoulli | 17.99 / 0.146 | 19.60 / 0.183 | 22.07 / 0.133 | 22.54 / 0.175 | 22.68 / 0.092 |
| gilbert_elliott | 16.89 / 0.185 | 18.09 / 0.210 | 19.66 / 0.174 | 20.32 / 0.204 | 20.41 / 0.119 |
| gop_chain | 15.96 / 0.240 | 16.87 / 0.258 | 17.52 / 0.257 | 17.99 / 0.279 | 18.08 / 0.204 |

## Gate G1 (pre-registered): multi-anchor vs two-anchor flow

Test split, gaps ≥ 4, paired over 11 clips: ΔPSNR = **-0.137 dB**, 95% CI [-0.375, +0.130], 4/11 clips improved, Wilcoxon p = 0.3203.

**G1 FAILED.** Reported as a negative result; two-anchor flow stays the classical default.

## Test split: each method vs RIFE (all gaps, paired over clips, Holm-corrected)

| Method | ΔPSNR vs RIFE | 95% CI | Clips better | Holm p |
|---|---:|---|---:|---:|
| hold | -1.25 | [-1.81, -0.77] | 0/11 | 0.0039 |
| linear | -0.38 | [-0.93, +0.07] | 5/11 | 0.6406 |
| flow | -0.28 | [-0.67, +0.02] | 4/11 | 0.6406 |
| flow_quadratic | -0.39 | [-0.77, -0.01] | 2/11 | 0.1611 |

## Test split: DISTS at 320×180, gaps 4 and 16 (lower is better)

| Gap | hold | linear | flow | flow_quadratic | rife |
|---|---:|---:|---:|---:|---:|
| 4 | 0.064 | 0.137 | 0.100 | 0.136 | 0.053 |
| 16 | 0.097 | 0.150 | 0.153 | 0.185 | 0.096 |

Run: 31 clips, 15065 frame-method rows, 70.6 min wall on 4 CPU workers. Error maps (80×45, float16) are stored outside git for C2/C3.
