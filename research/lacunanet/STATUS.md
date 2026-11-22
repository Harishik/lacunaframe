# LacunaNet status (2026-11-22)

Research checkpoint before GPU training. All numbers are on dev data only; no test data has been used for LacunaNet.

## What exists
- **LacunaNet v3**: a hybrid missing-frame reconstruction network. A parameter-free, differentiable pyramidal Lucas–Kanade layer (`classical_flow.py`) gives correct motion from step 0. Learned components (cost-volume flow residuals, flow and blend correction, half-resolution refinement, self-error head) are added behind ReZero gates. Motion runs through a physics trajectory (constant acceleration from outer anchors) and differentiable flow reversal. The model exports to ONNX and runs on CPU.
- **Motion-stratified training sampler** (`data.py`): it matches the training motion distribution to the target regime, which corrects a 4× motion mismatch in GOPRO.
- **Diagnosis trail** (`DESIGN_AUDIT.md`, sections 1–9): each design decision is tied to a measured failure and its fix, including the changes that were reverted.

## Current results (20 dev clips, 300 missing frames, paired 95% CIs clustered by clip)
| Comparison | Mean PSNR difference |
|---|---|
| v3 classical-prior pipeline vs linear blending | **+2.40 dB** [1.36, 3.62] |
| v3 after 1,000 CPU training steps vs its prior | −0.10 dB [−0.20, −0.02] (was about −1 dB before the fixes in sections 7–8) |
| CPU latency, v3 C=24 vs RIFE 4.9 (idle, interleaved) | **0.93×** |

## Not done yet
- GPU training (Colab T4, 60k iterations, batch 16) is ready (`colab_train.py`, `build_colab_cell.py`) and scheduled after the compute budget resets.
- After training: dev-only checkpoint selection (the step-0 prior is a candidate), then held-out benchmarks against RIFE/IFRNet/AMT on Vimeo-90K, SNU-FILM and Xiph, the latency gate (G9), the self-check (G11), and the conformal risk masks.
