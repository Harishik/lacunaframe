# Research context and implementation decisions

Literature reconnaissance: 2026-11-11, refreshed 2026-11-16 in [RELATED_WORK.md](RELATED_WORK.md). This is a focused survey of directly relevant primary pages and abstracts, not an exhaustive review or a reproduction of their experiments. The initial v0.1 used only classical baselines. v0.2 now incorporates pinned pretrained RIFE weights and attributed natural DAVIS footage; see the update below.

## Korea University PRISM Lab

The intended lab is [PRISM at Korea University](https://prism.korea.ac.kr/index.html), led by Hyungjin Chung, not the similarly named Kunsan lab. Its [research page](https://prism.korea.ac.kr/research.html) connects generative modeling, temporal understanding, inverse problems, and computational imaging. Its [publication archive](https://prism.korea.ac.kr/publications.html) provides recent context; dates and venue labels on a live archive are not independent verification of acceptance.

- **Diffusion Posterior Sampling for General Noisy Inverse Problems**, Chung et al., ICLR 2023. [Project](https://dps2022.github.io/diffusion-posterior-sampling-page/), [paper](https://arxiv.org/abs/2209.14687). Learned score priors plus measurement guidance motivate separating observations, corruption operators, and recovered estimates. LACUNAFRAME adopts that experimental separation, not DPS itself.
- **Accelerating Video Inverse Problem Solvers with Autoregressive Diffusion Models (AVIS)**, Kwon et al., 2026. [Paper](https://arxiv.org/abs/2605.20624). The abstract describes streaming autoregressive diffusion, measurement-consistent initialization, and a faster variant reducing consistency enforcement. Its GPU throughput figures are not comparable with our small CPU natural-video clips. We do not implement AVIS or claim to improve it.
- **ReDirector**: the lab research page links to [Creating Any-Length Video Retakes with Rotary Camera Encoding](https://byeongjun-park.github.io/ReDirector/). The linked project describes camera-conditioned retakes and rotary camera encoding. The search-indexed lab publication list instead displayed “Robust Video Diffusion with Reversible Frame Reordering.” These titles conflict; this document relies on the directly opened project page and does not infer that they describe the same algorithm. Camera-controlled generation is adjacent context, not missing-frame ground-truth recovery.
- **InvFusion: Bridging Supervised and Zero-shot Diffusion for Inverse Problems** (2025) and **ContextMRI: Enhancing Compressed Sensing MRI through Metadata Conditioning** (listed as 2027 in the current archive) appear in the archive. They suggest useful future questions about prior adaptation and conditioning, but their full methods were not assessed here and are not implemented.

## Related interpolation and classical foundations

- [RIFE: Real-Time Intermediate Flow Estimation](https://arxiv.org/abs/2011.06294), Huang et al., ECCV 2022: learned intermediate flow and arbitrary-time interpolation. Its IFNet is different from estimating endpoint flow and numerically inverting a scaled field.
- [FILM: Frame Interpolation for Large Motion](https://film-net.github.io/), Reda et al., ECCV 2022: multiscale learned motion and synthesis address large motion and disocclusion. These are precisely weak points of classical local optical flow; we do not claim parity.
- [OpenCV Farnebäck optical flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html): dense polynomial-expansion motion estimation. Implemented here using grayscale RGB conversions, pyramid scale 0.5, 3 levels, 21-pixel window, 4 iterations, poly_n 7, poly_sigma 1.5.
- [OpenCV geometric transforms](https://docs.opencv.org/4.10.0/da/d54/group__imgproc__transform.html): `remap` samples source coordinates for each destination pixel. Correct conventions matter more than calling a function named optical flow.
- [OpenCV inpainting](https://docs.opencv.org/4.x/df/d3d/tutorial_py_inpainting.html): Telea's fast-marching spatial repair is a separate baseline, not temporal reconstruction.

## What the prototype actually contributes

An inspectable educational experimental instrument: deterministic scenes, explicit absent samples, observation-only solvers, corruption-aware evaluation, failure diagnostics, and portable evidence. This is software integration, not demonstrated algorithmic novelty. No new hybrid learned model has been trained. v0.2 executes pretrained RIFE without fine-tuning. A defensible future research comparison would require matching observation operators, datasets, resolution, compute budget, and metrics, plus held-out natural videos, multiple seeds, ablations, uncertainty calibration, and the actual published implementations.

## Inverse formulation and flow conventions

Observations follow y = M B x + n: sample/mask operator M, optional blur B, additive noise n. Missing information admits many possible x. A preserved clean reference is evaluator-only; no restoration function receives it.

Nearest hold chooses the nearest observed time (earlier at ties). Linear blends observed bracketing frames with t=(i-left)/(right-left). Boundary gaps use nearest hold. Telea fills masks on observed frames and uses nearest hold for missing frames.

Farnebäck returns F_AB(p) with B correspondence p+F_AB(p); backward flow is F_BA. For an intermediate destination q, solve p+tF_AB(p)=q with five fixed-point steps. The second warp uses p+(1-t)F_BA(p)=q. Both remaps are destination-to-source maps. Reject paths outside image bounds or with |F_AB(p)+F_BA(p+F_AB(p))|≥1.5 pixels. Blend valid paths with temporal weights; if both are invalid, use ordinary linear blending. Report the fraction without two reliable paths. This is a consistency diagnostic, **not** calibrated uncertainty or an occlusion oracle. Fixed-point inversion may fail for strongly varying flow.

Brightness changes, large displacement, textureless surfaces, moving boundaries, and new content invisible in both anchors make recovery unreliable. Nonrigid motion and long gaps violate linear trajectories. Blur destroys high frequencies, and this release includes no deblurring or denoising. Preserve negative results instead of fabricating recovered detail.


## v0.2: pretrained inference is feasible on CPU

GPU training does not imply GPU-only deployment. We verified a pinned RIFE 4.9 ensemble ONNX export from [yuvraj108c/rife-onnx](https://huggingface.co/yuvraj108c/rife-onnx), upstream [Practical-RIFE](https://github.com/hzwer/Practical-RIFE), with ONNX Runtime CPUExecutionProvider. Inputs are RGB float32 NCHW in [0,1], zero-padded on right/bottom to multiples of 32; timestep is float32 [1]. Output is cropped and rounded to uint8. Arbitrary bracketing timesteps are passed directly to the network, not approximated with recursive midpoint blending. Endpoint times return exact input copies.

The default uses graph optimization and two threads, selected through actual CPU measurements. RIFE Fast halves input dimensions and resizes the estimate back; it preserves observed frames. This is a deliberate accuracy/compute tradeoff, **not distillation or pruning**. We have not quantized the model or assumed operator support for INT8. A future compression experiment should benchmark both operator execution and degradation on held-out footage; distillation additionally requires a training dataset, teacher outputs, compute, and clear train/test separation.

Natural evaluation uses DAVIS sequences described by Perazzi et al., [CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Perazzi_A_Benchmark_Dataset_CVPR_2016_paper.html). It is an adaptation of video-segmentation footage for controlled frame recovery, not the official DAVIS segmentation benchmark. Source mirror/encoding, hashes, licensing, source sample indices, and unknown checkpoint training overlap are disclosed. This establishes a working learned CPU path and measured improvements over our classical implementation on some conditions, not superiority over the lab's diffusion models.
