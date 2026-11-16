# A research claim must survive its strongest baseline

Project author: **Harishik Dev Singh Jamwal**.

Current status: engineering/reproducibility portfolio plus an exploratory CPU compression experiment. **No top-tier paper claim, new student model, distillation result, pruning result, or superiority to published competitors has been established.**

Q1 describes a journal's first quartile within a particular ranking/year/category; it is not a quality certificate for an individual manuscript. “Top 2%” is not a standard experimental acceptance criterion. A suitable research ambition is a credible, reproducible contribution that expert reviewers can test and potentially reject.

## The research question

Can a compact interpolation system improve the **quality–latency–memory frontier on real CPUs**, especially under long temporal gaps and changing visibility, relative to strong small interpolation models and compression baselines?

Scope the primary task to RGB video-frame interpolation. Spatial inpainting, denoising, deblurring, LLM inference and general diffusion inverse problems are separate tasks. They should not be combined into one leaderboard. LLM compression baselines are irrelevant unless a proposed method actually uses an LLM; adding one now would add cost and confound the claim.

CPU deployment is a worthwhile constraint. It is not by itself novelty. GPU-trained teachers can be used for CPU-deployed students; the paper must separately report training hardware, teacher-generation cost, student training cost, and inference hardware. No paid or GPU training run has been started here.

## Candidate hypothesis, not an asserted invention

**Gap-sensitive, operator-aware compression:** a student/precision allocation that protects motion/visibility information under increasing anchor separation might preserve reconstruction quality better than global low-bit quantization at matched measured CPU latency.

This needs a precise method beyond a list of standard techniques. PMQ-VE already addresses multi-frame quantization/distillation; ANVIL already examines quantized recurrent-flow failure; dynamic inference predates this project. The novelty boundary must be checked against their full methods and subsequent work before naming a new algorithm.

An initial technical path is to measure compression errors separately in flow/visibility features and synthesis features across gap lengths, then test whether a learned geometry-preservation objective or CPU-cost-constrained allocation fixes the observed failure. Preserve a full-precision teacher, train a structurally smaller student, use distillation only where its target is reliable, and validate the actual exported CPU graph. **These are proposed experiments; none of the training components is implemented yet.**

Do not assume every technique must be combined. Pruning, quantization and distillation need independent ablations; a redundant component should be removed if it adds complexity without a measured benefit.

## What has now been executed

The [compression pilot](results/compression-pilot/README.md) builds three quantized RIFE variants and compares them with FP32 4.9, half-resolution 4.9 and FP32 4.26. It uses a separate calibration scene, aligned actual pixels, fresh worker processes, saved predictions, measured quality/timing/memory, and diagnostic CPU operator traces.

The dynamic U8 artifact is roughly 3.46× smaller, yet it does not pass the preset 1.25× speed / ≤0.2 dB loss gate on both scenes. Neither static QDQ configuration passes. Operator traces show that much of the QDQ graph still executes ordinary `Conv`, alongside `QLinearConv` and dequantization operations. This supports an engineering conclusion—model bytes alone are inadequate—not a novel compression theorem or state-of-the-art result.

FP32 4.26 passes this small midpoint pilot's continuation gate. It lost to 4.9 in most of the earlier arbitrary-time corrupted-video conditions. That disagreement is a reminder to specify the task and evaluation distribution, not an excuse to choose whichever table is flattering.

## Evidence gates

| Gate | Work required | Current result |
|---|---|---|
| 0. Honest scope | Distinguish engineering integration, reproduced method, and new contribution | Documented. |
| 1. CPU feasibility | Executable compressed graphs, real quality and operator/memory/latency evidence | Six-config pilot complete; no PTQ speed winner across both scenes. |
| 2. Competitive baselines | Official small checkpoints/adapters, matching preprocessing, export parity, optimized runtime comparisons | Outstanding beyond RIFE-family variants. |
| 3. Independent evaluation | Untouched development/test scene manifests, official midpoint benchmarks and a separate missing-run task | Outstanding. Current clips were used in earlier exploratory work; unknown pretraining overlap. |
| 4. Actual learned contribution | Student architecture, legitimate train/validation split, checkpoints, training log, geometry/visibility objective or justified alternative | Not implemented. |
| 5. Controlled compression | Structural pruning rebuild, PTQ/QAT, teacher/student variants, comparable compute and independent ablations | PTQ controls only. |
| 6. Systems robustness | Warm/cold p50/p95, multiple fresh trials, full pipeline timings, CPU model/runtime/threads, RSS, energy only if measurable, sustained tests | Short pilot on one virtualized x86 host. Native 720p/1080p, ARM, sustained and energy measurements outstanding. |
| 7. Scientific robustness | Scene-level paired uncertainty, multiple training seeds, failure strata, negative cases, reproducible tables and plots | No generalization or significance claim. |
| 8. Manuscript | Related-work positioning, supported claims, actual ablation tables, limitations, artifact availability, target-venue requirements | Research protocol only; writing a results manuscript now would overstate progress. |

## What “outperforms” must mean

Choose the primary claim and budget before selecting a final method. Two defensible forms are:

1. Higher reconstruction/perceptual quality at the same measured CPU latency and memory budget than the strongest comparable small baseline.
2. Lower CPU latency and memory at matched quality, with a stated noninferiority margin and independent test scenes.

Show the entire Pareto frontier and the per-scene results. A model that is faster but less accurate is a tradeoff, not universal superiority. A CPU method beating a GPU paper's FPS at a different resolution, sampling protocol or hardware budget is not evidence of a win. Large models are quality references, not the only competition; include lightweight and already-distilled/pruned/quantized methods explicitly.

## Minimum evaluation design

- **Data:** training/development/test identities frozen separately; official Vimeo90K, SNU-FILM difficulty groups and another motion distribution subject to access/rights checks. Public-source pretraining overlap is recorded, not assumed absent. Current three DAVIS clips are diagnostics only.
- **Tasks:** standard midpoint triplets first; arbitrary times and missing runs separately. Anchor-only access for all solvers. Scene-cut and no-bracket cases have an explicit fallback/abstention protocol.
- **Quality:** exact official PSNR/SSIM conventions plus common RGB scores; LPIPS and an appropriate temporal/perceptual metric; occluded/motion-boundary/large-motion strata. Distinguish fidelity from plausibility and do not optimize only a convenient metric.
- **Compute:** same source arrays, output resolution, CPU thread budget, runtime/provider and precision; initialization/preprocessing/postprocessing/decode/encode scopes explicit. Report p50/p95, RSS, file bytes, parameters and sustained behavior. FLOPs are secondary, not a substitute for latency.
- **Statistics:** cluster by independent scene, not individual frames; paired intervals and multiple training seeds; no headline significance from the current two evaluation scenes or repeated timing calls.
- **Ablations:** same training budget for teacher/student and no-distillation control; pruning alone versus structural rebuild; quantization alone versus QAT; geometry objective on/off; fixed versus adaptive compute, including routing overhead; leave-one-gap regime out.
- **Reproducibility:** pinned source/checkpoint hashes, exported graphs, official-baseline parity tests, independent scoring scripts, complete failures and unsuccessful runs. No hidden target access or test-driven selection.

Relevant venues to study include IEEE TIP, TCSVT and TPAMI; fit and submission requirements matter more than marketing a percentile. No acceptance or ranking outcome can be guaranteed in advance.

## Decisions from this phase

Keep the existing workbench as the experiment instrument. Keep the current main application honest. Pursue a CPU research branch with a measurable hypothesis and the gates above. Do not label the app “distilled,” “pruned,” “novel,” “Q1-level” or “better than competitors” until the corresponding evidence exists. The next engineering milestone is official lightweight baseline parity and a frozen independent dataset—not another dashboard redesign or a speculative paper title.
