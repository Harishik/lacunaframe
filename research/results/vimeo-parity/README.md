# Vimeo90K T1 parity check (448×256 midpoint)

Per-image RGB PSNR, averaged. Published numbers are for the full 3,782-triplet test set and other checkpoints, so they are context, not a matched comparison. Timings were taken while another benchmark shared the CPU and are indicative only.

## shard 0 only (first quarter of the test set, ordered by sequence) (256 triplets)

| Method | PSNR (dB) | SSIM | s / frame (CPU) |
|---|---:|---:|---:|
| linear | 25.79 | 0.7625 | 0.123 |
| flow | 31.62 | 0.9231 | 0.232 |
| rife | 33.88 | 0.9540 | 0.864 |

## 64 evenly spaced triplets from each of the 4 shards (256 triplets)

| Method | PSNR (dB) | SSIM | s / frame (CPU) |
|---|---:|---:|---:|
| linear | 25.47 | 0.7586 | 0.146 |
| flow | 31.08 | 0.9175 | 0.504 |
| rife | 33.48 | 0.9525 | 1.093 |

Published, full test set: RIFE (ECCV 2022, paper Table) 35.62 dB; IFRNet-S (CVPR 2022, paper) 35.59 dB; AMT-S (CVPR 2023, paper) 35.97 dB.

## Open issue: RIFE scores about 2 dB below the published number

Our pipeline scores 33.48 dB on the shard-balanced sample (33.88 on shard 0) against the 35.62 dB reported for the ECCV 2022 RIFE checkpoint. Frame order was checked: endpoint PSNR is lower than neighbour PSNR. **Parity is not claimed.**

| Hypothesis | Test | Status |
|---|---|---|
| Subset bias (shard 0 only) | Re-sample 64 triplets from each of the 4 shards | **Refuted**: the balanced sample is 0.4 dB *lower* |
| Checkpoint: Practical-RIFE 4.9 ensemble is not the paper's Vimeo-trained model | Run the paper checkpoint through the same adapter | Open (needs a one-off ONNX export) |
| Mirror re-encoding | Compare mirror PNG bytes with the official test release | Open (official server returned 403) |
| Adapter convention (input order, timestep) | Symmetry check f(a,b,½) vs f(b,a,½) on 32 balanced triplets | **Consistent**: 33.70 vs 33.71 dB, forward/reverse agreement 40.8 dB. Padding/scale are still only testable through paper-checkpoint parity |

Until resolved, RIFE numbers in LacunaBench are internal comparisons only.
