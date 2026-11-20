# Protocol v2: confirmation on untouched data, external baselines, revised G4

Frozen 2026-11-20, **after** the v1 results and **before** any v2 data is downloaded or any v2 analysis is run. v1 stays the record of what was pre-registered and what happened (G1 failed, G2 passed, G3 passed, G4 failed). v2 does not reinterpret v1. It adds a fresh test set and re-specifies one gate, with the reason stated. Machine-readable copy: [recoverability-v2.json](recoverability-v2.json).

## 1. Why v2

1. **The v1 test split has been used.** Every new claim needs data that no model, threshold or design choice has seen.
2. **G4 was uninformative under a base-rate shift.** With τ fixed at 20, 44% of test pixels were "bad", against 18% on dev, so any mask meeting 90% coverage flagged about 82% of pixels. v2 fixes τ from dev statistics *before* seeing the new test set.
3. **Only RIFE-family models had been compared.** v2 adds two published small interpolators with their official weights.

## 2. New confirmatory test set: Xiph 1080p (untouched)

Twelve sequences from the Xiph derf collection (`media.xiph.org/video/derf/y4m/`), fixed now: crowd_run, park_joy, ducks_take_off, in_to_tree and old_town_cross (1080p50); riverbed, pedestrian_area, rush_hour, station2, sunflower, tractor and blue_sky (1080p25).

- **Selection rule (fixed before download):** a 48-frame window centred in each sequence, at stride 2 for 50 fps sources and stride 1 for 25 fps sources (25 fps effective). Exactly those frames are fetched with HTTP range requests.
- **Format:** YUV 4:2:0 converted to RGB with OpenCV's I420 conversion, then the same 16:9 centre crop and INTER_AREA resize to 640×360 as v1.
- **Tasks:** identical to v1 (gaps 1–32, Bernoulli / Gilbert–Elliott / GOP-chain at 25%), with seeds derived from clip names.
- **Use:** confirmation only. Nothing is fit, tuned or calibrated on Xiph.

Storage: each clip's window is about 150 MB transient and about 25 MB stored. The project stays under the 6 GB cap.

## 3. Methods added

| Method | Source | Licence | Export check |
|---|---|---|---|
| AMT-S (CVPR 2023) | `huggingface.co/lalala125/AMT` (`amt-s.pth`, linked from the official README) | CC BY-NC-SA 4.0 | ONNX vs PyTorch output, max abs error < 1e-3 on real frames |
| IFRNet-S and IFRNet (CVPR 2022) | Official Dropbox archive from the IFRNet README | MIT | same |
| RIFE, paper model (ECCV 2022) | Google Drive link "RIFE model reported by our paper" from the official README | MIT | same; used for the Vimeo90K parity question only |

All run on CPU through ONNX Runtime with the same padding conventions as the official inference code. If an export fails its parity check, that method is reported as *not evaluated*, never replaced.

## 4. Gates (judged on Xiph only)

| Gate | Pass condition |
|---|---|
| **G2′** recoverability transfer | The dev-trained RIFE per-frame model (unchanged from v1) has pooled Spearman ρ ≥ 0.6 on Xiph, and beats the pair-only baseline with a paired per-clip CI excluding 0 |
| **G3′** calibration transfer | Using the v1 dev-calibrated λ (Mondrian, short/long), coverage ≥ 0.88 in every Xiph stratum with ≥ 3 clips containing bad pixels |
| **G4′** conditioning (revised) | τ′ = the RIFE pixel-error value at which 20% of **dev** pixels are bad (computed from dev only and recorded before Xiph is scored). At α = 0.1, the conditioned mask's flagged area ≤ 0.85 × the pair-only area on Xiph, with both coverages ≥ 0.88 |
| **P1** export parity | Each ONNX export matches its PyTorch reference (max abs error < 1e-3 on 8 real frame pairs) |
| **P2** Vimeo checkpoint question | The paper RIFE model on the shard-balanced Vimeo subset scores ≥ 35.0 dB. Pass: the 2 dB gap is attributed to the checkpoint. Fail: the gap stays open |

Baseline comparisons (AMT, IFRNet vs RIFE 4.9 and flow) are **descriptive**: scene-clustered paired CIs, Holm-corrected, no gate.

## 5. Unchanged from v1

Unit of analysis (the clip), bootstrap CIs, paired Wilcoxon with Holm correction, PSNR cap of 60 dB, LPIPS on all missing frames, the error-map format, the storage rules, and failure retention.
