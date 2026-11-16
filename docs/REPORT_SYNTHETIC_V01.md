# v0.1 verification report

Measured on this CPU execution environment, 2026-11-11. Three synthetic scenarios × four methods. Exact parameters and per-frame values are in each scenario report.

| Scenario | Method | All-pixel PSNR (dB) | Affected-pixel PSNR (dB) | SSIM | Algorithm time (s) |
|---|---|---:|---:|---:|---:|
| 1 | hold | 35.187 | 26.825 | 0.99634 | 0.0036 |
| 1 | linear | 37.964 | 29.603 | 0.99605 | 0.0247 |
| 1 | flow | 46.770 | 38.409 | 0.99520 | 0.4407 |
| 1 | telea | 35.187 | 26.825 | 0.99634 | 0.0052 |
| 2 | hold | 26.931 | 21.258 | 0.98048 | 0.0025 |
| 2 | linear | 28.338 | 22.665 | 0.97934 | 0.0058 |
| 2 | flow | 27.949 | 22.276 | 0.97095 | 0.5988 |
| 2 | telea | 26.931 | 21.258 | 0.98048 | 0.0027 |
| 3 | hold | 20.277 | 12.288 | 0.85731 | 0.0025 |
| 3 | linear | 20.374 | 12.385 | 0.86249 | 0.0047 |
| 3 | flow | 20.453 | 12.464 | 0.88366 | 0.1739 |
| 3 | telea | 28.334 | 20.346 | 0.89496 | 0.0779 |

## Interpretation

Flow improves PSNR on the short-gap orbital scene. On the crossing-object long gap, linear blending has higher PSNR than flow. Telea performs best on the spatial-mask scenario, where temporal baselines intentionally leave masks unrepaired. Flow does not win every metric: on scenario 1 its PSNR advantage does not translate into the highest SSIM. All-frame scores are diluted by unchanged observations; use the affected-pixel and affected-frame breakdowns.

These are three controlled examples with one seed, not a statistical benchmark or evidence of superiority to RIFE, FILM, AVIS, or diffusion restoration. No learned model was run.

## Verification

- pytest: 10 tests covering interpolation, forward/backward flow direction and translated midpoint, alignment, seeded corruption, affected-pixel metrics, identical/small/NaN handling, Telea preservation, API end-to-end, invalid input, PNG sequences, actual MP4 upload, GIF fallback, and TTL cleanup.
- Production build, strict TypeScript check, and ESLint pass.
- Playwright: real browser experiment, frame navigation, error-map view, report retrieval, desktop screenshot and mobile no-horizontal-overflow check.
- Three real benchmark runs produce CSV, JSON, original/damaged/reconstructed MP4 files, error plots, and frame PNGs. Media decoded to verify all 48 frames.
- FastAPI TestClient emits a third-party Starlette/httpx deprecation warning; tests still pass. Docker image recipe is provided but not built in this session.

## Remaining limits

Synthetic scenes only for the saved benchmark; no natural-video performance claim. Nominal-FPS normalization of uploads, no audio, no learned diffusion, no denoising/deblurring, no freehand mask editor (procedural brush mask only), heuristic flow confidence, and no authenticated multi-user deployment. Session history is browser-memory only; artifacts expire after one hour. The error map requires an untouched reference and cannot measure fidelity to an unknown real-world scene.
