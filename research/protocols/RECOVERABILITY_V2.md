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

## Amendment 1 — 2026-11-20 (before any arbitrary-time checkpoint is run)

**Finding that triggered it.** On the v1 test split, the Vimeo90K-trained AMT-S, IFRNet-S and IFRNet trail RIFE 4.9 by −0.1 dB near t = 0.5 but by −3.4 to −3.8 dB near t = 0.1/0.9. Two-anchor flow's deficit stays flat (≈ −0.5 dB). Vimeo90K triplets only ever contain t = 0.5, so these checkpoints are effectively midpoint models, and ranking them on arbitrary-time gaps would be unfair to the methods.

**Change.** Each baseline is also evaluated with its *official arbitrary-time checkpoint* trained for ×8 interpolation on GoPro: `gopro_amt-s.pth` (`huggingface.co/lalala125/AMT`), and `IFRNet_GoPro.pth` and `IFRNet_S_GoPro.pth` (official IFRNet archive). The same export-parity gate P1 applies. Both checkpoint families are reported, labelled `*_vimeo` (midpoint-trained) and `*_gopro` (arbitrary-time).

**Caveat recorded in advance.** The GoPro checkpoints were trained on the GOPRO training split. SNU-FILM's GoPro test clips come from GOPRO *test* sequences: different scenes, same camera and capture pipeline. Results on the SNU test split may therefore favour the `*_gopro` checkpoints. The Xiph set has no such overlap and is the fair comparison.

## Results against the v2 gates (2026-11-20)

| Gate | Outcome | Evidence |
|---|---|---|
| **P1** export parity | **Passed** for all 7 exports (≤ 3.9e-5) | `$LACUNA_EXTERNAL/weights/baselines-manifest.json` |
| **P2** Vimeo checkpoint | **Passed**: paper RIFE 35.40 dB; AMT-S/IFRNet-S/IFRNet within 0.13–0.22 dB of published | [vimeo-parity](../results/vimeo-parity/README.md) |
| **G2′** recoverability transfer | **Passed**: ρ = 0.881 on untouched Xiph (pair-only 0.845; per-clip Δρ CI [+0.021, +0.153]) | [confirmation-v2](../results/confirmation-v2/README.md) |
| **G3′** calibration transfer | **Failed**: worst stratum coverage 0.815 (4 of 11 gated strata below 0.88) | same |
| **G4′** conditioning | **Failed**: area ratio 0.961 (pass ≤ 0.85) | same |

**What this establishes.** Recoverability prediction (C2) generalises to a new camera, new content and a new frame rate without retraining; this is the project's strongest result. Conformal coverage (C3) holds within the calibration source (G3, v1) but **not** across this dataset shift: split-conformal guarantees assume exchangeable calibration and test clips, and Xiph violates that. Making C3 a contribution requires shift-aware calibration (e.g. weighted conformal prediction with recoverability-based density-ratio weights, or a small labelled calibration set from the target domain). That is the next research step, to be pre-registered before it is run.
