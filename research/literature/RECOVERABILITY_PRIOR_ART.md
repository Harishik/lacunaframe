# Recoverability and calibrated missing-frame recovery: prior-art audit

Reviewed 2026-11-20 (Phase 0 of the recoverability pivot). Author: **Harishik Dev Singh Jamwal**.

Scope: does a paper already deliver (a) a *recoverability* estimate for missing video frames made from observations only, (b) *distribution-free calibrated* per-pixel risk for reconstructed missing frames, or (c) a missing-frame benchmark with realistic loss processes? Eight web searches across uncertainty in VFI, conformal risk control for restoration, conformal methods under temporal dependence, long-gap benchmarks, hallucination detection in restoration, and selective restoration. Review depth is abstract/snippet level unless stated. **Search absence does not establish novelty.** A full-text pass on the "must read" rows is required before committing to a paper title.

Machine-readable ledger: [recoverability-reading-log.json](recoverability-reading-log.json).

## 1. What already exists

### Uncertainty in video frame interpolation (closest technical neighbours)

| Work | What it does | What it does not do | Status |
|---|---|---|---|
| Plack et al., *Frame Interpolation Transformer and Uncertainty Guidance*, CVPR 2023 | Predicts an uncertainty map for interpolated frames (rendered content) and uses it to guide rendering of extra frames | No distribution-free guarantee; two-anchor midpoint setting; not missing runs | Must read (seen only as a citation; venue to verify) |
| *Time-adaptive VFI based on Residual Diffusion*, arXiv 2504.05402 (2025) | Uses diffusion stochasticity for pixel-wise uncertainty on interpolated frames (animation) | Uncalibrated sample spread; no coverage claim; animation domain | Must read |
| *Multiple Hypothesis Flow Estimation for VFI under Matching Ambiguity*, arXiv 2608.07120 (2026) | Several flow hypotheses + reliability router; MA-HD ambiguity benchmark | The router picks one hypothesis; reliability is not calibrated risk | Must read; MA-HD is a candidate test set |
| *Disambiguation for VFI*, arXiv 2311.08007 | Reduces speed/direction ambiguity with distance maps and step splitting | Treats uncertainty as error to remove, not an output | Background |
| VFI-DPA, arXiv 2304.12664 | Predicts pair difficulty from the two inputs, routes to fast/heavy models; difficulty-labelled dataset | Coarse difficulty classes for routing; no per-pixel risk, no gap-length model, no guarantee | Must read; **primary baseline for C2** |

### Distribution-free risk for image restoration (closest statistical neighbours)

| Work | What it does | Gap relative to us |
|---|---|---|
| Angelopoulos et al., *Image-to-Image Regression with Distribution-Free UQ* (2022) | Calibrated per-pixel intervals for image-to-image models | Single images; exchangeable calibration |
| Conformal Risk Control, arXiv 2208.02814 | Controls the expected value of any monotone loss | General framework we would build on |
| Teneggi et al., *How to Trust Your Diffusion Model*, arXiv 2302.03791 | High-dimensional CRC minimising interval length (denoising, CT) | Images; i.i.d. samples |
| Kutiel et al., *Conformal Prediction Masks* (2023) and conformal SR with guarantees | Mask so masked divergence stays below a risk level | Closest *mask* mechanism; images only |
| CHEM, arXiv 2512.09806 | Conformalised quantile regression hallucination measure for colourisation/SR/inpainting | Images, not temporal |
| Conformal Bounds on Full-Reference IQA, arXiv 2505.09528 | Conformal bounds on PSNR/SSIM-type quality for inverse problems | Explicitly marginal (not conditional) guarantee; images |
| SafeRestore, arXiv 2609.03475 | Selective restoration: return measured image, one of five restorations, or send to human review; binomial risk certificates | Industrial stills; *decision*-level, close in spirit to our abstention protocol |
| Selective CRC, arXiv 2512.12844; Conditional Conformal Risk Adaptation, arXiv 2504.07611; non-monotone CRC, arXiv 2604.01502 | Extensions we may need (abstention, conditional coverage, non-monotone loss) | Methodological tools, not video |

### Conformal methods under temporal dependence

| Work | Relevance |
|---|---|
| CUPS, arXiv 2412.10431 | Conformal pose-shape from video; uses Barber et al. (2023) non-exchangeable weighting. Closest precedent for handling frame dependence |
| Non-exchangeable CP for temporal GNNs, arXiv 2507.02151 (KDD 2025) | Formalises non-exchangeability over time |
| Why split conformal works under temporal dependence, arXiv 2510.02471 | Theory we can cite for scene-blocked splits |
| Conformal coverage for video temporal grounding, arXiv 2608.07434 | Conformal on video, but segment-level, not dense pixels — must read to rule out overlap |
| C³, arXiv 2512.05927 | Calibrated dense uncertainty for controllable video *generation* via a latent probe (proper scoring rules, ECE). Not conformal, not restoration of observed footage |

### Missing-run / long-gap benchmarks

| Work | Relevance |
|---|---|
| ArbInterp + MultiInterpBench, ICLR 2026 | Multi-ratio (2×–32×) generative interpolation benchmark. Uniform ratios, no realistic loss process, no recoverability labels. **Must compare against** |
| *Benchmarking VFI*, arXiv 2403.17128 | Consistent metrics and per-attribute (motion magnitude) breakdowns; synthetic test set |
| TAI video frame inpainting, arXiv 1803.07218 | Early contiguous missing-run reconstruction |
| BSCV (NTU) | Realistic *bitstream* corruption; notes interpolation fails on large motion/spread errors. Closest to our realistic-loss idea; must read |
| LongE2V, arXiv 2607.08770 | Long-horizon interpolation with event cameras (extra sensor; different contract) |

## 2. Novelty verdict

| Proposed contribution | Verdict | Why |
|---|---|---|
| **C1 LacunaBench** (realistic burst/codec loss, gap 1→32, strata labels, frozen splits, sealed-target protocol) | **Open, but must differentiate from MultiInterpBench and BSCV** | Neither combines realistic loss *processes* with recoverability/strata labels and a sealed target contract. Defensible if the loss models and labels are the point, not the dataset size |
| **C2 recoverability index** (pre-reconstruction, gap-aware, per-pixel, method-agnostic) | **Open; VFI-DPA is the baseline to beat** | VFI-DPA predicts coarse pair difficulty for routing. No found work predicts per-pixel expected error across gap lengths and transfers across reconstructors |
| **C3 conformal risk for missing-frame reconstruction** | **Open but at risk of "conformal applied to a new domain"** | Image CRC (Angelopoulos, Teneggi, Kutiel, CHEM) and VFI uncertainty (Plack, residual diffusion) both exist. The contribution must therefore be the *video-specific* problem: temporal non-exchangeability, shift across gap lengths and loss processes, and risk defined over missing runs rather than single frames |

**Decision: continue with the pivot, with C3 reframed.** The publishable core is C2 + C3 *together*: a recoverability score that conditions the calibration (so guarantees adapt to gap length and visibility) and remains valid under distribution shift across loss processes. "We added conformal prediction to VFI" alone would likely be judged incremental.

Kill criteria carried into Phase 1–3:
1. If a full read of Plack et al., 2504.05402 or 2608.07434 shows calibrated (coverage-tested) risk for missing video frames, drop C3 as a headline and lead with C1 + C2.
2. If VFI-DPA-style difficulty features already reach Spearman ρ ≥ 0.6 with per-frame error on held-out data, C2 must show a clear margin or per-pixel value to remain a contribution.
3. If MultiInterpBench or BSCV already provide burst/codec loss with sealed targets, C1 becomes an extension, not a contribution.

## 3. Must-read list (full text, before Phase 1 design freeze)

1. Plack et al., CVPR 2023 — uncertainty guidance in VFI
2. arXiv 2504.05402 — residual-diffusion VFI uncertainty
3. arXiv 2608.07120 — multi-hypothesis flow + MA-HD
4. arXiv 2304.12664 — VFI-DPA
5. arXiv 2608.07434 — conformal coverage for video
6. ArbInterp / MultiInterpBench (ICLR 2026)
7. BSCV — bitstream-corruption benchmark
8. Kutiel et al. 2023 and Teneggi et al. 2023 — mask/interval CRC mechanics
9. CUPS 2412.10431 — non-exchangeable calibration on video

## Sources

[CUPS](https://arxiv.org/pdf/2412.10431) · [C³](https://arxiv.org/html/2512.05927) · [VFI-DPA](https://ar5iv.labs.arxiv.org/html/2304.12664) · [Benchmarking VFI](https://bohrium.dp.tech/paper/arxiv/2403.17128) · [TAI](https://ar5iv.labs.arxiv.org/html/1803.07218) · [BSCV](https://dr.ntu.edu.sg/server/api/core/bitstreams/3018cecb-61d0-4e43-9006-bcc5af1884ab/content) · [Residual-diffusion VFI](https://arxiv.org/abs/2504.05402) · [Disambiguation for VFI](https://arxiv.org/html/2311.08007v3) · [Multi-hypothesis flow](https://arxiv.org/pdf/2608.07120) · [Conformal Risk Control](https://arxiv.org/html/2208.02814v4) · [How to Trust Your Diffusion Model](https://arxiv.org/html/2302.03791) · [Conformal IQA bounds](https://arxiv.org/html/2505.09528) · [SR with conformal guarantees](https://www.researchgate.net/publication/389055709_Image_Super-Resolution_with_Guarantees_via_Conformal_Generative_Models) · [SafeRestore](https://arxiv.org/abs/2609.03475) · [CHEM](https://arxiv.org/html/2512.09806v2) · [Selective CRC](https://arxiv.org/pdf/2512.12844) · [Conditional Conformal Risk Adaptation](https://arxiv.org/pdf/2504.07611) · [Non-monotone CRC](https://arxiv.org/pdf/2604.01502) · [Non-exchangeable CP for temporal GNNs](https://arxiv.org/pdf/2507.02151) · [Split conformal under temporal dependence](https://arxiv.org/pdf/2510.02471) · [Conformal video temporal grounding](https://arxiv.org/pdf/2608.07434) · [ArbInterp / MultiInterpBench](https://proceedings.iclr.cc/paper_files/paper/2026/file/36572cc8bb2a939f0d998847cb44776e-Paper-Conference.pdf) · [LongE2V](https://arxiv.org/pdf/2607.08770) · [AceVFI survey](https://arxiv.org/pdf/2506.01061)
