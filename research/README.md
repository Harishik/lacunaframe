# CPU efficiency research track

Author: **Harishik Dev Singh Jamwal**.

[Local CPU resumption handoff (2026-11-17)](LOCAL_CPU_HANDOFF.md) records the
Intel MacBook hardware inspection, local execution blockers, and next steps.
Read it before rerunning the archived pilot on this laptop.

This branch begins the research program requested after the v0.3 portfolio release. It adds **actual compressed CPU graphs and measurements**, while keeping a clear distinction between a working instrument and a new research result. The requested top-tier research standard has **not yet been achieved**.

| Open | What it contains |
|---|---|
| [Research program](RESEARCH_PROGRAM.md) | Candidate hypothesis, direct novelty collisions, fair comparison contract, concrete evidence gates and outstanding training/evaluation work |
| [Prior-art audit](literature/PRIOR_ART.md) | Relevant lightweight, distillation, quantization, dynamic-compute and diffusion work; review depth and execution status |
| [Measured pilot](results/compression-pilot/README.md) | Six executed CPU configurations, scene-disjoint calibration, per-triplet results and a quality/latency plot |
| [Fixed pilot protocol](protocols/compression-pilot.json) | Case selection, thread budget, timing scope, predeclared continuation criterion and exclusions |
| [Pilot runner](scripts/compression_pilot.py) | Fresh-process execution, actual quantization, saved predictions, hashes, logs and summary |
| [Operator profiler](scripts/profile_operators.py) | Raw ORT traces and explicit scope of operator-duration summaries |
| [Evidence checks](tests/test_compression_evidence.py) | Recomputed scores from saved pixels, calibration/target alignment, actual quantized CPU inference and trace provenance |

![Measured CPU compression feasibility](results/compression-pilot/quality-latency.png)

Reproduce from the repository root:

```bash
.venv/bin/python research/scripts/compression_pilot.py
.venv/bin/python research/scripts/profile_operators.py
.venv/bin/python research/scripts/summarize_pilot.py
.venv/bin/python -m pytest -q backend/tests research/tests
```

**Verification:** 21 backend/research pytest tests pass, including rescoring all 108 saved prediction triplets and executing all three quantized graphs on CPU. [Captured output](results/compression-pilot/pytest.txt).

The existing virtual environment's lock includes ONNX/ORT. The runners overwrite this exploratory pilot directory; preserve previous trials in a separate copy. No training, distillation, structural pruning, API spending or hosted compute is performed. GPU training is a possible future resource choice; CPU deployment does not require CPU-only training, but any additional paid compute remains unauthorized.

Quantized model binaries inherit RIFE's MIT attribution. Saved DAVIS prediction arrays/figures retain CC BY-NC 4.0. [Asset provenance](../assets/THIRD_PARTY.md) · [Original portfolio documentation](../docs/README.md).
