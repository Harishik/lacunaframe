# Changes

## 0.3.1 (unreleased) — Phase 0: portability, prior art, presentation — 2026-11-20

- **Fixed a cross-platform reproducibility flaw.** OpenCV decoded the source MP4s with FFmpeg on Linux but AVFoundation on macOS, and the two differ by up to 71 levels per pixel. All decoding now goes through the pinned FFmpeg path in `backend/decode.py`, which reproduces the archived scores bit-exactly. See `research/results/portability/`.
- Found and documented that static all-Conv INT8 (QDQ) outputs differ across ONNX Runtime versions and CPUs (35 dB vs archive). The evidence test now records drift outside the archived runtime instead of demanding bit-equality.
- Research runners: `--out` / `LACUNA_PILOT_OUT` for fresh trial directories, with the archived pilot protected from overwrites. Fixed macOS peak-RSS units (bytes, not KiB) in the pilot and the profiler. Added macOS CPU, memory and power provenance. Added `onnx` to the requirements, plus an Intel-macOS lock (`requirements-lock-macos-x86_64.txt`; ORT 1.31 has no Intel-macOS wheel).
- Upload decoding derives the container suffix from the validated MIME type.
- Added a recoverability and calibrated-recovery prior-art audit with a machine-readable ledger and novelty verdict.
- Rewrote the README to lead with results. Added `CITATION.cff` and a Hugging Face Docker Space scaffold (`deploy/huggingface/`, not deployed), plus optional same-origin frontend serving (`LACUNA_STATIC`).
- 25 backend/research tests pass on Intel macOS (ORT 1.23.2).

## Documentation and private repository edition — 2026-11-16

- Added a visual README and structured reading routes, setup/workflow/protocol/architecture guides, and archive indexes.
- Added an offline interactive field guide with an actual reference/reconstruction wipe, measured chart controls, reduced-motion support and copyable setup commands.
- Added reproducible real-pixel GIFs, a failure contact sheet, error timeline, measured comparison charts and visual-provenance checks.
- Preserved earlier synthetic experiments and screenshots in labeled archive folders; included both model exports, source clips, lossless arrays and every existing experiment.
- Added documentation link/data validation and an offline responsive/accessibility browser test.
- Changed hosted CI to manual dispatch and added a publishing script that requires private visibility. The owner subsequently created the repo; the verification definition is now kept as an inactive template outside `.github/workflows` because integration access to Actions settings is restricted.

## 0.3.0 — 2026-11-16

- Replaced the frontend with the reconstruction desk: source bin, footage viewer, damage/restore inspector, dedicated evaluation and editorial case study.
- Added responsive layouts, local typography, frame selection on the error plot, reliable form labels and shortcut help, explicit offline behavior and automated accessibility checks.
- Added a pinned RIFE 4.26 CPU research adapter and 36 matched checkpoint runs. Retained 4.9 as default because 4.26 lost affected PSNR in 16/18 conditions despite faster inference.
- Added lossless benchmark packs with evaluator-only targets, exact timestamp validation and artifact hashes; included an executed export/restoration/evaluation example.
- Added current primary-source research comparison covering PRISM, AVIS, InverseCrafter, EPS, InvFusion, BiM-VFI, controllable tracking-based interpolation and FILM.
- Added artifact verification, browser-managed development servers, CI checks, and a build-verified non-root Docker backend.

## 0.2.0

Natural DAVIS footage, RIFE 4.9 ONNX CPU inference and half-resolution variant, 90 measured natural method runs, CPU profiling, provenance and licenses.

## 0.1.0

Classical pixel baselines, reproducible synthetic scenes, corruption/evaluation pipeline, FastAPI API and initial interface.
