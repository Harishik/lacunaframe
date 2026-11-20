# Published baselines on the LacunaBench v1 test split (11 GoPro clips)

AMT-S, IFRNet-S and IFRNet use official weights exported to ONNX (export parity ≤ 3.9e-5). Plain names are the **Vimeo90K-trained (midpoint) checkpoints**, which reproduce their published Vimeo90K PSNR within 0.2 dB (see `../vimeo-parity/`). `*_gopro` are the official **arbitrary-time GoPro ×8 checkpoints** added by protocol v2 amendment 1. Caveat: they were trained on GOPRO train scenes, and this test split uses GOPRO test scenes from the same camera. All methods received identical observations. The clip is the unit. Comparisons are descriptive (protocol v2).

## PSNR (dB) by gap length

| Gap | hold | linear | flow | flow_quadratic | rife | amt_s | ifrnet_s | ifrnet | amt_s_gopro | ifrnet_s_gopro | ifrnet_gopro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18.03 | 19.98 | 22.91 | 23.25 | 23.70 | 23.85 | 23.45 | 23.81 | **23.93** | 23.71 | 23.78 |
| 2 | 17.86 | 19.03 | 21.35 | 21.85 | 22.21 | 19.41 | 19.46 | 19.47 | 22.07 | 22.21 | **22.22** |
| 4 | 16.67 | 17.70 | 18.73 | 19.11 | 19.55 | 17.61 | 17.43 | 17.39 | **19.89** | 19.65 | 19.46 |
| 8 | 16.19 | 17.07 | 17.29 | 17.54 | 17.80 | 16.36 | 16.12 | 16.10 | **18.03** | 17.89 | 17.91 |
| 16 | 14.65 | 15.50 | 15.44 | 15.30 | 15.76 | 14.92 | 14.84 | 14.81 | 15.73 | **15.98** | 15.84 |
| 32 | 13.75 | 14.57 | 14.36 | 14.06 | 14.45 | 14.08 | 14.13 | 14.11 | 14.60 | **14.67** | 14.52 |

## LPIPS by gap length (lower is better)

| Gap | hold | linear | flow | flow_quadratic | rife | amt_s | ifrnet_s | ifrnet | amt_s_gopro | ifrnet_s_gopro | ifrnet_gopro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.147 | 0.184 | 0.122 | 0.161 | 0.084 | 0.077 | 0.078 | **0.076** | 0.181 | 0.137 | 0.131 |
| 2 | 0.138 | 0.174 | 0.134 | 0.163 | **0.099** | 0.120 | 0.118 | 0.117 | 0.188 | 0.144 | 0.139 |
| 4 | 0.192 | 0.217 | 0.199 | 0.242 | **0.137** | 0.227 | 0.204 | 0.191 | 0.224 | 0.191 | 0.187 |
| 8 | 0.231 | 0.251 | 0.257 | 0.287 | **0.191** | 0.323 | 0.304 | 0.284 | 0.288 | 0.266 | 0.258 |
| 16 | 0.311 | 0.326 | 0.351 | 0.385 | **0.264** | 0.453 | 0.437 | 0.427 | 0.390 | 0.375 | 0.366 |
| 32 | 0.386 | 0.392 | 0.423 | 0.470 | **0.361** | 0.511 | 0.507 | 0.511 | 0.479 | 0.470 | 0.461 |

## Loss processes at 25% (PSNR dB / LPIPS)

| Process | hold | linear | flow | flow_quadratic | rife | amt_s | ifrnet_s | ifrnet | amt_s_gopro | ifrnet_s_gopro | ifrnet_gopro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bernoulli | 17.99 / 0.146 | 19.60 / 0.183 | 22.07 / 0.133 | 22.54 / 0.175 | 22.68 / 0.092 | 21.83 / 0.109 | 21.81 / 0.103 | 21.79 / 0.100 | 22.99 / 0.182 | 22.85 / 0.143 | 22.82 / 0.136 |
| gilbert_elliott | 16.89 / 0.185 | 18.09 / 0.210 | 19.66 / 0.174 | 20.32 / 0.204 | 20.41 / 0.119 | 18.61 / 0.183 | 18.51 / 0.170 | 18.50 / 0.160 | 20.64 / 0.207 | 20.33 / 0.181 | 20.36 / 0.171 |
| gop_chain | 15.96 / 0.240 | 16.87 / 0.258 | 17.52 / 0.257 | 17.99 / 0.279 | 18.08 / 0.204 | 16.69 / 0.305 | 16.55 / 0.285 | 16.53 / 0.273 | 18.18 / 0.284 | 18.09 / 0.268 | 18.00 / 0.260 |

## ΔPSNR vs RIFE 4.9, all contiguous gaps (paired over clips, Holm-corrected)

| Method | Δ (dB) | 95% CI | Clips better | Holm p |
|---|---:|---|---:|---:|
| hold | -1.25 | [-1.81, -0.77] | 0/11 | 0.0098 |
| linear | -0.38 | [-0.93, +0.07] | 5/11 | 0.9609 |
| flow | -0.28 | [-0.67, +0.02] | 4/11 | 0.9609 |
| flow_quadratic | -0.39 | [-0.77, -0.01] | 2/11 | 0.3223 |
| amt_s | -0.79 | [-1.27, -0.37] | 1/11 | 0.0205 |
| ifrnet_s | -0.84 | [-1.34, -0.39] | 1/11 | 0.0176 |
| ifrnet | -0.85 | [-1.33, -0.43] | 1/11 | 0.0176 |
| amt_s_gopro | +0.12 | [-0.03, +0.24] | 9/11 | 0.4150 |
| ifrnet_s_gopro | +0.19 | [-0.03, +0.37] | 9/11 | 0.6992 |
| ifrnet_gopro | +0.07 | [-0.12, +0.23] | 8/11 | 0.9609 |

## ΔPSNR vs RIFE 4.9, long gaps (≥ 8) (paired over clips, Holm-corrected)

| Method | Δ (dB) | 95% CI | Clips better | Holm p |
|---|---:|---|---:|---:|
| hold | -0.94 | [-1.44, -0.52] | 1/11 | 0.0195 |
| linear | -0.11 | [-0.60, +0.28] | 7/11 | 1.0000 |
| flow | -0.22 | [-0.60, +0.09] | 6/11 | 1.0000 |
| flow_quadratic | -0.39 | [-0.77, -0.01] | 3/11 | 0.4980 |
| amt_s | -0.66 | [-1.12, -0.25] | 2/11 | 0.1230 |
| ifrnet_s | -0.69 | [-1.18, -0.24] | 4/11 | 0.3760 |
| ifrnet | -0.71 | [-1.17, -0.28] | 3/11 | 0.1953 |
| amt_s_gopro | +0.11 | [-0.06, +0.25] | 8/11 | 0.5078 |
| ifrnet_s_gopro | +0.20 | [-0.03, +0.40] | 9/11 | 0.6992 |
| ifrnet_gopro | +0.08 | [-0.12, +0.26] | 8/11 | 0.9609 |

CPU seconds per missing frame at 640×360 (indicative; runs shared the CPU): hold 0.00, linear 0.02, flow 0.54, flow_quadratic 0.38, rife 1.32, amt_s 3.06, ifrnet_s 1.46, ifrnet 2.08, amt_s_gopro 2.43, ifrnet_s_gopro 1.17, ifrnet_gopro 1.62.

## Reading these results

- **Midpoint-trained checkpoints are not arbitrary-time models.** The Vimeo90K weights trail RIFE 4.9 by about 0.1 dB near t = 0.5 but by up to 3.8 dB near t = 0.1/0.9 (deficit grows monotonically with |t − 0.5|). That is why gap ≥ 2 rows collapse for `amt_s`, `ifrnet_s` and `ifrnet`.
- **The arbitrary-time GoPro checkpoints fix that**, and on this split edge RIFE 4.9 in PSNR on long gaps (+0.08 to +0.20 dB, CIs include 0). They are blurrier: LPIPS is worse at every gap ≥ 2. Caveat (protocol v2 amendment 1): they were trained on GOPRO training scenes, and this split uses GOPRO test scenes from the same camera.
- **The fair comparison is Xiph** (no overlap; see `../confirmation-v2/`). There the GoPro checkpoints tie RIFE 4.9 on PSNR and lose clearly on LPIPS.
