# Recoverability (C2) and calibrated risk masks (C3): v1 results

Models are fit on dev (20 YouTube clips) and evaluated once on test (11 GoPro clips). Inputs come from observed frames only. The pair-only baseline sees the two anchor frames but not the gap length, the timestep or the outer anchors (a feature-based analogue of VFI-DPA's pair-difficulty idea, not its CNN). Clips are the unit for CIs.

## C2: per-frame recoverability (Spearman ρ between predicted and actual PSNR on test)

| Target method | Full model ρ (pooled) | Pair-only ρ (pooled) | Full ρ per clip, mean [95% CI] | Δρ vs pair-only per clip [95% CI] | Clips better |
|---|---:|---:|---|---|---:|
| hold | 0.802 | 0.618 | 0.794 [0.676, 0.879] | +0.274 [+0.212, +0.346] | 11/11 |
| linear | 0.799 | 0.667 | 0.789 [0.675, 0.875] | +0.143 [+0.080, +0.207] | 9/11 |
| flow | 0.745 | 0.675 | 0.808 [0.715, 0.881] | +0.078 [+0.062, +0.095] | 11/11 |
| flow_quadratic | 0.827 | 0.717 | 0.904 [0.864, 0.937] | +0.127 [+0.094, +0.168] | 11/11 |
| rife | 0.675 | 0.610 | 0.761 [0.696, 0.822] | +0.079 [+0.040, +0.111] | 10/11 |

**Gate G2 (pre-registered, judged on RIFE, the strongest reconstructor): PASSED** (pooled ρ = 0.675; per-clip Δρ vs pair-only CI [+0.040, +0.111]).

## C2 transfer across reconstructors (pooled Spearman ρ on test)

| Trained on → evaluated on | hold | linear | flow | flow_quadratic | rife |
|---|---:|---:|---:|---:|---:|
| flow | 0.717 | 0.748 | 0.745 | 0.832 | 0.673 |
| rife | 0.706 | 0.740 | 0.742 | 0.828 | 0.675 |

## C2 per-pixel and C3 conformal risk masks

A pixel is bad when its pooled mean absolute error exceeds 20/255. Masks are calibrated with conformal risk control at α = 0.1 (target: miss at most 10% of bad pixels), with clips as the exchangeable units (20 dev clips; finite-sample term 1/21). Area is the fraction of missing-frame pixels flagged; smaller is better at equal coverage.

| Method | Score | Pixel AUROC (bad vs good) | Test coverage of bad pixels | Flagged area | Area vs pair-only |
|---|---|---:|---:|---:|---:|
| flow | conditioned (full) | 0.824 | 0.974 | 0.820 | -0.0% |
| flow | pair-only | 0.785 | 0.954 | 0.820 | — |
| flow | anchor difference (heuristic) | 0.773 | 0.987 | 0.920 | +12.2% |
| rife | conditioned (full) | 0.774 | 0.971 | 0.826 | +0.7% |
| rife | pair-only | 0.743 | 0.951 | 0.820 | — |
| rife | anchor difference (heuristic) | 0.727 | 0.991 | 0.946 | +15.4% |

**Gate G4 (RIFE): FAILED**: the conditioned mask flags 0.826 of pixels vs 0.820 for pair-only (-0.7% smaller), with test coverage 0.971 vs 0.951.

## C3 coverage by stratum (conditioned score, Mondrian by gap length, RIFE)

| Stratum | Coverage | Flagged area | Clips with bad pixels |
|---|---:|---:|---:|
| gap1 / short (≤4) | 0.970 | 0.616 | 11 |
| gap2 / short (≤4) | 0.899 | 0.609 | 11 |
| gap4 / short (≤4) | 0.979 | 0.816 | 11 |
| bernoulli / short (≤4) | 0.927 | 0.615 | 11 |
| gilbert_elliott / short (≤4) | 0.932 | 0.717 | 9 |
| gop_chain / short (≤4) | 0.986 | 0.753 | 7 |
| gap8 / long (>4) | 0.978 | 0.835 | 11 |
| gap16 / long (>4) | 0.994 | 0.950 | 11 |
| gap32 / long (>4) | 0.998 | 0.980 | 11 |
| gilbert_elliott / long (>4) | 0.968 | 0.780 | 7 |
| gop_chain / long (>4) | 0.944 | 0.790 | 11 |

**Gate G3 (RIFE): PASSED**: worst stratum coverage 0.899 (target ≥ 0.88). Strata with fewer than 3 test clips containing bad pixels are shown but not gated.

Caveats: 11 test clips give wide uncertainty on coverage. The guarantee is marginal over clips, not per frame. τ and α were fixed in this script before the test split was evaluated.

## Why G4 failed (post-hoc diagnosis, does not change the verdict)

With τ fixed at 20/255, **43.9% of RIFE's test pixels are bad, against 18.3% on dev** (per-clip range 27–62% on test vs 2–44% on dev). The GoPro clips contain much larger motion than the YouTube clips. Reaching 90% coverage of bad pixels therefore forces any score to flag most of each frame. Both masks flag about 82% of pixels and over-cover (0.95–0.97), because a λ calibrated on dev is conservative under this shift. The conditioned score does rank pixels better (AUROC 0.774 vs 0.743 for RIFE, 0.824 vs 0.785 for flow), but at this base rate better ranking barely changes the area needed for the target coverage.

Proposed for protocol v2 (to be frozen *before* rerunning, not applied to these results):
1. Define "bad" relative to each frame's achievable quality (for example, the top error quantile within each gap stratum) or report area over a grid of τ, so masks remain informative under base-rate shift.
2. Calibrate on a dev set whose motion distribution brackets the test distribution, or use weighted conformal calibration with recoverability-based covariate-shift weights (as in Cover and CUPS).
3. Report area at matched *realised* coverage as a secondary diagnostic, clearly separated from the gated result.

## Gate summary

| Gate | Result | Evidence |
|---|---|---|
| G2 recoverability (RIFE) | **Passed** | Pooled per-frame ρ = 0.675 ≥ 0.6. Beats pair-only in 10/11 test clips, per-clip Δρ CI [+0.040, +0.111]. Transfers across reconstructors (flow-trained model ranks RIFE errors at ρ = 0.673) |
| G3 calibration (RIFE) | **Passed** | Worst gated stratum coverage 0.899 ≥ 0.88 across gap lengths and loss processes |
| G4 conditioning (RIFE) | **Failed** | Area 0.826 vs 0.820 (pair-only); masks uninformative at a 44% test bad-pixel rate (see diagnosis) |
