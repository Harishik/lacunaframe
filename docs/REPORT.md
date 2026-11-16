# v0.3 — the reconstruction desk and measured model comparison

The v0.3 release redesigns the working interface and adds a recent-checkpoint CPU comparison and lossless external evaluation contract. The original natural-video study below is preserved. Actual pretrained inference on CPU; no new model training or claimed algorithmic novelty.

## Natural-video results

Three real DAVIS sequences × two gap regimes × seeds 7/42/99 = 18 conditions, five methods each (90 outputs). Results are arithmetic means of per-experiment affected-pixel scores, with nine runs per row.

| Regime | Method | Affected PSNR ↑ (dB) | Affected SSIM ↑ | Seconds / missing frame ↓ |
|---|---|---:|---:|---:|
| scattered | hold | 17.297 | 0.4350 | 0.0005 |
| scattered | linear | 18.652 | 0.4612 | 0.0012 |
| scattered | flow | 19.513 | 0.5448 | 0.0382 |
| scattered | rife | 20.793 | 0.6461 | 0.1414 |
| scattered | rife_fast | 20.333 | 0.5620 | 0.0574 |
| long-gap | hold | 16.384 | 0.3989 | 0.0001 |
| long-gap | linear | 17.527 | 0.4184 | 0.0005 |
| long-gap | flow | 17.605 | 0.4353 | 0.0377 |
| long-gap | rife | 17.881 | 0.4853 | 0.1328 |
| long-gap | rife_fast | 17.746 | 0.4238 | 0.0570 |

RIFE exceeds classical flow in affected PSNR in **12/18 conditions**. It is not universally better. Its mean advantage is about 1.28 dB on scattered loss and 0.28 dB on long gaps. RIFE Fast is about 2.4× faster on this host, with lower average PSNR and SSIM. Six-frame gaps at 12fps are half a second of absent observations; shape, visibility and trajectory ambiguity remain substantial.

## CPU optimization experiment

Separate process per setting, same real frame pair, one warmup and seven timed calls. Graph optimizations and thread settings change execution, not the checkpoint. No pruning, quantization, distillation or GPU was used.

| Threads | Graph optimization | Median inference (s) | Process peak RSS (MiB) |
|---:|---|---:|---:|
| 1 | disabled | 0.2194 | 169.8 |
| 1 | all | 0.1952 | 212.4 |
| 2 | all | 0.1127 | 213.5 |
| 4 | all | 0.3222 | 214.0 |

Two threads with graph optimization were fastest among the tested settings. Four threads were slower on this container; hardware-dependent threading matters. Inference is not claimed to be universally real time. Full reconstruction wall time includes metrics and encoding beyond these timings.

## Visible evidence

- [BMX long-gap comparison](../examples/natural-v02/natural-bmx-trees-long-gap-42/comparison.png): same worst RIFE missing frame for all methods; shows a genuine failure, not a curated success.
- [Breakdance scattered-loss comparison](../examples/natural-v02/natural-breakdance-scattered-42/comparison.png).
- [Machine-readable complete results](../examples/natural-v02/summary.json) and [CSV](../examples/natural-v02/summary.csv).
- Each condition contains damaged video, five restored videos, full JSON/CSV and model/source provenance. Original downloaded footage is in `assets/video/`.

## Verification and scope

All 108 natural benchmark MP4s were decoded and verified against their report frame counts.

17 pytest tests pass, including actual natural-video upload, real RIFE CPU inference at nonmultiple dimensions, arbitrary timesteps, deterministic output, observed-pixel preservation, no dependence on missing storage placeholders, model-unavailable error, API/media path, GIF fallback, cleanup, classical motion direction, and metrics. Strict TypeScript/build/lint and Playwright exercise the learned natural-video UI. See the final verification output for run times.

References are decoded compressed MP4 mirror frames, not camera RAW. No training or fine-tuning was performed; checkpoint training overlap with DAVIS is unknown. Three clips are inadequate to establish broad performance or statistical significance; repeated seeds share the same scenes. No claim is made against PRISM diffusion systems, whose code/weights were not benchmarked. RIFE is a pretrained interpolation model, not a general noise/blur/occlusion remover. Natural-media licenses are CC BY-NC 4.0 per the pinned mirror, separate from our MIT code.

## v0.3 checkpoint comparison

A pinned RIFE 4.26 vs-mlrt ONNX export runs successfully on CPU with the correct 11-channel coordinate/time ABI. In a new matched 36-run comparison, 4.26 loses affected PSNR to 4.9 in 16/18 conditions. Scattered-loss means are 20.488 versus 20.793 dB; long-gap means are 17.415 versus 17.881 dB. 4.26 is approximately 1.6× faster on this host. The 4.9 configuration is an ensemble; this is a comparison of deployable exports, not a version-only ablation. The default remains 4.9; the candidate and its negative results are retained as opt-in research evidence.

[CSV](../examples/checkpoint-v03/summary.csv) · [JSON](../examples/checkpoint-v03/summary.json) · [Research landscape](RELATED_WORK.md). Twelve candidate-study preview clips and eighteen per-condition reports are included. The lossless external evaluation contract also completed a real export → linear restoration → timestamp-checked scoring path in `examples/external-pack-v03/`.

## Interface release

The new frontend separates Workspace, Evaluation and The project, with modular typed components, local typography, real media, synchronized viewing, frame-selectable plots, session history, mobile source navigation and explicit engine-offline states. Browser screenshots are regenerated from actual runs in [docs/design](design/). The project page remains readable when the backend is absent; processing controls explicitly require the backend.

## Final verification, 2026-11-16

- `pytest -q`: **17 passed**, one upstream Starlette/httpx deprecation warning.
- Production Vite build, ESLint and strict TypeScript checks pass.
- Playwright: **4 passed (25.4s)**, including actual CPU restoration/media/report downloads, mobile/keyboard/config invalidation, engine-offline behavior, and Axe WCAG A/AA checks on all three pages after a completed run.
- Artifact audit: **5 source/model checksums, 120 fully decoded MP4s, 36 condition reports**, matching frame counts and summary scores.
- Docker image **build passed** with the optional host trust-bundle secret. Container startup hit the workspace disk cap in Docker's VFS storage driver, so container runtime verification remains incomplete. The local image and its dependency cache were removed to recover space.

A verification-log file initially caused Vite to reload active browser experiments. Generated evidence/backend paths are now excluded from the frontend watcher; the final complete browser run passes with that fix. Captured command output is in [verification](verification/). No paid services, deployment or GitHub push were performed.

## Private repository documentation edition

Structured reading routes and archive indexes now accompany an offline interactive field guide. Figures/GIFs are generated from the actual lossless BMX observations and measured summaries. Local verification passes **17 pytest tests**, **5 browser tests (31.4s)**, production build, ESLint and strict TypeScript. The added browser test checks the guide with zero HTTP requests, actual chart values, a working reference/recovery wipe, mobile overflow, reduced-motion alternatives and Axe A/AA accessibility. `scripts/check_documentation.py` validates links and embedded evidence; earlier synthetic runs and interface captures remain in the archive.
