# Vimeo90K T1 parity check (448×256 midpoint)

Per-image RGB PSNR, averaged. Published numbers are for the full 3,782-triplet test set, so they are context rather than a matched comparison; see the resolved section below for the checkpoint-matched result. Timings were taken while another benchmark shared the CPU and are indicative only.

## 64 evenly spaced triplets from each of the 4 shards (256 triplets)

| Method | PSNR (dB) | SSIM | s / frame (CPU) |
|---|---:|---:|---:|
| linear | 25.47 | 0.7586 | 0.061 |
| flow | 31.08 | 0.9175 | 0.232 |
| rife | 33.48 | 0.9525 | 0.621 |
| rife_paper | 35.40 | 0.9652 | 0.647 |
| amt_s | 35.84 | 0.9693 | 1.953 |
| ifrnet_s | 35.43 | 0.9660 | 0.901 |
| ifrnet | 35.64 | 0.9679 | 1.083 |

Published, full test set: RIFE (ECCV 2022, paper Table) 35.62 dB; IFRNet-S (CVPR 2022, paper) 35.59 dB; IFRNet (CVPR 2022, paper) 35.80 dB; AMT-S (CVPR 2023, paper) 35.97 dB.

## Resolved (protocol v2, gate P2): the gap was the checkpoint, and the pipeline reproduces published numbers

| Model (official weights, ONNX on CPU) | Ours, balanced 256 (dB) | Published, full 3,782 (dB) | Difference |
|---|---:|---:|---:|
| RIFE, ECCV 2022 paper model | 35.40 | 35.62 | −0.22 |
| IFRNet-S | 35.43 | 35.59 | −0.16 |
| IFRNet | 35.64 | 35.80 | −0.16 |
| AMT-S | 35.84 | 35.97 | −0.13 |
| RIFE 4.9 (Practical-RIFE, workbench default) | 33.48 | — | — |

**P2 passed** (paper RIFE ≥ 35.0 dB). Four independent official checkpoints land within 0.13–0.22 dB of their published full-test-set numbers on our 256-triplet sample. That confirms the data path (mirror PNGs, frame order, colour), the padding conventions and the PSNR convention. The earlier 2 dB gap belongs to **Practical-RIFE 4.9**, a model tuned for practical footage rather than Vimeo90K, and not to the pipeline. Every export passed gate P1 (ONNX vs PyTorch ≤ 3.9e-5).

History of this issue: the shard-0 subset gave RIFE 4.9 33.88 dB; the shard-balanced re-sample gave 33.48 dB (subset bias refuted); a symmetry check passed (33.70 vs 33.71 dB); and the paper checkpoint resolved it.
