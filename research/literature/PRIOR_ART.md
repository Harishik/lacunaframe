# Efficiency research: what is already known

Reviewed 2026-11-16. This is a focused primary-source audit, not a completed systematic review of all VFI research. Search phrases covered efficient/lightweight VFI, CPU deployment, quantization, distillation, structured pruning, adaptive computation, and temporal-gap robustness. The linked AceVFI survey provides a wider starting bibliography. Search absence does not establish novelty.

[Machine-readable review ledger](reading-log.json) records review depth, source links and execution status.

## Directly relevant prior art

| Work / primary source | What blocks a simplistic novelty claim | Review / execution status |
|---|---|---|
| [RIFE, ECCV 2022](https://arxiv.org/abs/2011.06294) and [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) | Efficient learned interpolation and distillation are established. Running pretrained weights in ORT is deployment integration. | Existing 4.9/4.26 CPU adapters executed; official paper architecture is not reproduced exactly. |
| [IFRNet, CVPR 2022](https://arxiv.org/abs/2205.14620), [official implementation](https://github.com/ltkong218/IFRNet) | Compact joint flow/feature refinement, task-oriented flow distillation and feature geometry consistency already address efficiency. | Abstract and official implementation documentation reviewed; checkpoint not executed here. Small/Base variants belong in the comparison set. |
| [AMT, CVPR 2023](https://arxiv.org/abs/2304.09790), [official implementation](https://github.com/MCG-NKU/AMT) | Efficient convolutional correlation/multi-field synthesis competes against larger transformer models. | Abstract and official README/method overview reviewed; AMT-S is not executed here. |
| [Motion-Aware Dynamic Architecture for Efficient Frame Interpolation, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Choi_Motion-Aware_Dynamic_Architecture_for_Efficient_Frame_Interpolation_ICCV_2021_paper.pdf) | Local motion-dependent resolution/depth allocation predates this project. “Use more compute for difficult motion” alone is not new. | Primary search-indexed paper excerpts reviewed; direct PDF access returned 403, so a complete method review is pending. |
| [PMQ-VE, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/bcfdaf04b54a69f47623c973c864ee8d-Abstract-Conference.html), [official repository](https://github.com/xiaoBIGfeng/PMQ-VE) | Video-aware low-bit calibration and progressive multi-teacher distillation already exist, including a VFI evaluation. | Proceedings, PDF method/experiments and official repository checked. Repository currently advertises forthcoming code/weights, so no executed reproduction is claimed. |
| [ANVIL, 2026 v3](https://arxiv.org/abs/2603.26835v3), [full text](https://arxiv.org/html/2603.26835v3) | Operator incompatibility, quantized flow-state accumulation and codec-assisted residual architectures are already analyzed for efficient deployment. | Abstract and relevant full-text deployment/quantization sections reviewed. A **preprint submitted to TCSVT**, not a verified accepted journal result. Mobile NPU + codec-vector inputs differ from our RGB-only CPU task. |
| [Hierarchical Flow Diffusion, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Hai_Hierarchical_Flow_Diffusion_for_Efficient_Frame_Interpolation_CVPR_2025_paper.html) | Efficient diffusion interpolation is also an existing direction. | Primary indexed abstract reviewed; direct page access returned 403; full reproduction pending. |
| [SpeedVFI, ICML 2026](https://proceedings.mlr.press/v306/ding26t.html) | Sequence-level one-step diffusion and trajectory distillation already reduce sampling overhead. | Official proceedings abstract reviewed; no CPU implementation benchmarked. |
| [RDVFI, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1a1f7aeb1f9bd9052cfd11181430b806-Abstract-Conference.html) | Low-resolution trajectory generation separated from high-resolution rendering is already an efficiency strategy. | Official proceedings abstract reviewed; advertised FPS is not our hardware-matched measurement. |
| [AceVFI, TCSVT 2026 / arXiv v3](https://arxiv.org/abs/2506.01061v3), [bibliography](https://github.com/CMLab-Korea/Awesome-Video-Frame-Interpolation) | A survey covering 250+ papers documents a broad existing field. A few handpicked comparisons cannot establish novelty. | Abstract, journal metadata and bibliography reviewed; not every cited work has been read. |
| [ONNX Runtime quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) | QDQ, dynamic quantization and selective node quantization are standard tools. | Actual dynamic U8, Conv QDQ and trunk QDQ graphs built and executed in the new pilot. |

## The most consequential distinctions

**PMQ-VE is particularly close to the proposed generic compression claim.** Its method uses a coarse-to-fine quantization process and progressive supervision across precision levels. The paper's VFI experiments use EMA-VFI variants; its reported quality convention is luminance-channel PSNR/SSIM. It also evaluates perceptual metrics and reports training/fine-tuning on eight V100 GPUs. Those are training resources, not proof that inference must use a GPU. Our RGB PSNR, RIFE checkpoint, small DAVIS subset and ORT CPU timing are not directly comparable with that table.

**ANVIL changes the information and hardware contract.** It uses decoder-exposed motion vectors and a heterogeneous mobile pipeline with NPU inference. Its useful lesson is to optimize actual operators and deployment behavior. Its reported latency cannot be ranked against our two-thread x86 CPU measurements, and a codec-assisted model cannot silently receive additional side information in an RGB-only benchmark.

**Dynamic routing needs a stronger claim than “skip easy regions.”** Existing dynamic architecture work already motivates motion-dependent allocation. A new router would need a demonstrably different risk/budget mechanism, overhead-inclusive CPU measurements, and a matched comparison with fixed-compute and existing dynamic baselines.

**Parameter count, serialized bytes, FLOPs and CPU latency are different quantities.** Half-resolution inference leaves checkpoint parameters unchanged. Unstructured zeroing of weights does not remove dense convolution work. Quantized weights can still execute through floating-point kernels after graph transformations. The new operator profiles make that distinction visible.

## Comparison roster and present gaps

| Tier | Required examples | Present status |
|---|---|---|
| Classical sanity controls | nearest, linear, bidirectional flow | Already implemented; keep these but do not use them as the only competitors. |
| Small learned interpolation | IFRNet-S/Base, AMT-S, an appropriate small RIFE configuration | RIFE family only. Other official small checkpoints/adapters and native-runtime parity are outstanding. |
| Compression methods | standard PTQ/QAT, structured pruning, student distillation; PMQ-VE if released/reproducible | Three standard PTQ configurations executed. No trained student, QAT or structural pruning. |
| Strong larger quality references | suitable larger interpolation checkpoint; at least one relevant generative method if resources permit | RIFE ensemble is not a substitute for this whole tier. No diffusion competitor executed. |
| Deployment baselines | ORT and at least one applicable optimized CPU runtime, with matching model/precision | ORT only. ncnn projects illustrate deployment options; **Vulkan must not be counted as CPU-only execution**. |

Use official task splits and each implementation's exact color/metric convention, then provide a clearly labeled common RGB comparison. Separate midpoint interpolation from arbitrary-time interpolation and contiguous missing-run reconstruction. If a method is midpoint-only, evaluate it directly on the common midpoint task; report any recursive arbitrary-time extension as a separate adaptation.

The [earlier research context](../../docs/RELATED_WORK.md) covers PRISM-related inverse problems, FILM, BiM-VFI and controllable tracking-based interpolation. These sources remain relevant, but efficient VFI/quantization is the closer novelty neighborhood for the requested CPU research program.
