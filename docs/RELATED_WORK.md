# Research landscape and comparison contract

Reviewed 2026-11-16. Primary sources: the Korea University PRISM research page, its JavaScript publication archive, linked paper abstracts, and official project pages. This is a focused implementation survey, not an exhaustive literature review or an independent reproduction of the cited results. Paper descriptions below are attributed claims; the only directly measured model comparison here is our own CPU experiment.

## What recent work demonstrates

| Work / source | Relevant capability | Relation to this implementation |
|---|---|---|
| [PRISM: research](https://prism.korea.ac.kr/research.html) / [publication archive](https://prism.korea.ac.kr/publications-data.js) | Measurement-aware inverse problems, generative priors, video reasoning; the archive lists 2025–2027 work | Motivates preserving the observation operator and evaluating recovered hypotheses. LACUNAFRAME is independent, with no lab affiliation. Future venue labels are reported by the lab, not evidence of a finished independent review. |
| [AVIS, Kwon et al., 2026, v2](https://arxiv.org/abs/2605.20624v2) | Streaming autoregressive video diffusion with measurement-consistent initialization; Flash reduces repeated consistency enforcement | Broader video inverse solver, not simply two-frame interpolation. The abstract's RTX 4090 throughput cannot be ranked against our 320×180 CPU timings. We do not implement AVIS. |
| [InverseCrafter, Hong et al., ECCV 2026](https://arxiv.org/abs/2512.05672v2) | Latent-domain masking for video recapture/inpainting while preserving a pretrained VDM | Shows why a faithful forward operator matters. Its pretrained generative backbone and novel-view task differ from our pixel-domain baselines. Not implemented. |
| [Exact Posterior Score, Mammadov et al., 2026](https://arxiv.org/abs/2606.17048) | Operator-dependent denoising training objective for linear Gaussian inverse problems | Relevant to a future degradation-aware model. It requires training/fine-tuning; our flow consistency diagnostic is not a posterior or calibrated uncertainty. |
| [InvFusion, Elata et al., 2025](https://arxiv.org/abs/2504.01689v2) | Integrates the degradation operator into a trained diffusion denoiser | Architectural inspiration for a future hybrid, not something obtained by combining two existing output images. The paper evaluates FFHQ/ImageNet, not this missing-frame protocol. |
| [BiM-VFI, Seo et al., CVPR 2025](https://kaist-viclab.github.io/BiM-VFI_site/) | Bidirectional motion fields for nonuniform trajectories, content-aware upsampling, VFI-centric teacher supervision | A direct future interpolation comparator. Its LPIPS/STLPIPS claims cannot be converted into our PSNR gains. We have not executed its checkpoint. |
| [Controllable Tracking-Based VFI, Briedis et al., SIGGRAPH 2025](https://studios.disneyresearch.com/2025/07/17/controllable-tracking-based-video-frame-interpolation/) | Whole-sequence sparse tracks, editable trajectories, inference-time hallucination/blur control | A concrete capability beyond this app: expert trajectory correction. Our brush mask is a corruption operator, not a motion editor. We link the publication rather than redistribute its copyrighted media. |
| [FILM, Reda et al., ECCV 2022](https://film-net.github.io/) | Multiscale learned motion/synthesis for large displacement and disocclusions | Older but relevant quality reference. This release does not claim to match FILM on Xiph or other official benchmarks. |
| [RIFE, Huang et al., ECCV 2022](https://arxiv.org/abs/2011.06294) / [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) | Learned intermediate flow and arbitrary-time synthesis | Actually executed: pinned 4.9 ensemble and optional 4.26 ONNX exports on CPU. Practical checkpoints are distinct from reproducing the original paper's official settings. |

PRISM's archive also lists ReDirector, ReDesign, Video Parallel Scaling, and ContextMRI. They cover adjacent generation, design decomposition, video understanding, and medical imaging rather than the same controlled frame-recovery task. ReDirector's archive title and linked project title differ; we do not merge them into one asserted algorithm. ContextMRI is labeled 2027 in the current archive. We do not claim to have implemented or comprehensively reviewed every lab publication.

## The newer-checkpoint experiment

GPU-trained weights can run on CPU. This release verifies that directly rather than excluding the possibility. RIFE 4.26 uses an 11-channel vs-mlrt input ABI: two RGB frames, time, normalized x/y coordinates and their pixel-step scales. Coordinates refer to the padded 64-multiple dimensions. The adapter, hashes, source revisions, per-condition reports and output clips are included.

In the identical 18 conditions, 4.26 has lower affected PSNR in **16/18** cases. Equal-weight means:

| Regime | RIFE 4.9 PSNR / SSIM | RIFE 4.26 PSNR / SSIM | Seconds per absent frame: 4.9 / 4.26 |
|---|---|---|---|
| Scattered | 20.793 / 0.6461 | 20.488 / 0.6379 | 0.1285 / 0.0800 |
| Long gap | 17.881 / 0.4853 | 17.415 / 0.4604 | 0.1255 / 0.0794 |

Source: [`examples/checkpoint-v03/summary.json`](../examples/checkpoint-v03/summary.json). Warm inference, same host, two CPU threads; timing is host-specific and includes boundary holds. The 4.9 export is an ensemble while 4.26 is not, so this compares deployable configurations rather than isolating architecture version. The candidate was selected after viewing the original study: this is exploratory, not a new held-out test. 4.9 remains the UI default; 4.26 is an opt-in research adapter and reproducible experiment, not an advertised universal improvement.

No student has been trained. ONNX graph optimization and half-resolution inference are implemented. Distillation, structured pruning and INT8 quantization remain possible engineering experiments, but need a training/calibration split, supported operators, measured accuracy and end-to-end CPU profiling. Merely exporting a model is not distillation.

## How to make a valid outside comparison

Use `scripts/benchmark_pack.py` to export lossless observations and a separate evaluator-only target file. External inference receives `inputs.npz`, **never `targets.npz`**. Return RGB uint8 NHWC frames and copy the exact timestamp array. The evaluator rejects changed dimensions, shifted timestamps, and implicit float rescaling. It hashes both artifacts and uses the same affected-support metrics as the app.

```bash
.venv/bin/python scripts/benchmark_pack.py export --source natural-bmx-trees --regime long-gap --seed 42 --out /tmp/lacuna-comparison
# Run the outside model on inputs.npz and write predictions.npz with frames + timestamps.
.venv/bin/python scripts/benchmark_pack.py evaluate --targets /tmp/lacuna-comparison/targets.npz --predictions /tmp/lacuna-comparison/predictions.npz --method 'model name + checkpoint hash' --out /tmp/lacuna-comparison/result.json
```

Do not compare publication headline numbers with these rows. Match resolution, color range, source stride, observations, gap locations, model weights and precision, output timestamps, CPU threads, warm/cold runtime, and evaluation support. Record dataset provenance and possible training overlap. Perceptual metrics such as LPIPS/STLPIPS are not currently calculated. For an algorithmic claim, add independently held-out scenes and published task protocols, paired comparisons, ablations, and compute-normalized evaluations. Three clips with repeated seeds do not establish state of the art.

## Presentation lessons applied

The official FILM and BiM-VFI project pages lead with visible evidence and connect it to methods and experiments. This release gives the comparison viewer priority, separates experiment setup from result interpretation, and provides an editorial project page with real output media, measured claims, sources, and failure cases. This is an independently authored layout; no project template, screenshots, or research result figures were copied.
