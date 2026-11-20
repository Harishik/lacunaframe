# Natural-v02 re-analysis: scene-clustered statistics

Unit of analysis: scene (3 DAVIS clips); seeds 7/42/99 are averaged within each scene. 95% percentile bootstrap over scenes (10,000 resamples). Paired differences are against two-anchor optical flow.

## scattered

| Method | Affected PSNR (scene mean) | 95% CI | Δ vs flow | 95% CI of Δ | Scenes won | Holm p | CI excludes 0 |
|---|---:|---|---:|---|---:|---:|---|
| hold | 17.30 | [14.96, 18.73] | -2.22 | [-3.74, -1.38] | 0/3 | 1.00 | yes |
| linear | 18.65 | [16.26, 20.11] | -0.86 | [-2.36, -0.00] | 0/3 | 1.00 | yes |
| flow | 19.51 | [16.49, 22.46] | — | — | — | — | — |
| rife | 20.79 | [19.13, 24.02] | +1.28 | [-0.36, +2.64] | 2/3 | 1.00 | no |
| rife_fast | 20.33 | [18.48, 22.77] | +0.82 | [-1.11, +3.26] | 2/3 | 1.00 | no |

## long-gap

| Method | Affected PSNR (scene mean) | 95% CI | Δ vs flow | 95% CI of Δ | Scenes won | Holm p | CI excludes 0 |
|---|---:|---|---:|---|---:|---:|---|
| hold | 16.38 | [13.58, 18.19] | -1.22 | [-1.40, -1.11] | 0/3 | 1.00 | yes |
| linear | 17.53 | [14.74, 19.11] | -0.08 | [-0.47, +0.19] | 2/3 | 1.00 | no |
| flow | 17.61 | [14.69, 19.58] | — | — | — | — | — |
| rife | 17.88 | [15.07, 20.19] | +0.28 | [-0.17, +0.60] | 2/3 | 1.00 | no |
| rife_fast | 17.75 | [15.26, 20.09] | +0.14 | [-0.66, +0.57] | 2/3 | 1.00 | no |

## Reading this honestly

With n = 3 scenes, the smallest attainable two-sided Wilcoxon p-value is 0.25, and a bootstrap over three values can only resample those three numbers. These intervals describe *these three clips*; they are not population inferences. Treat this as the template the v1 protocol will fill with many more scenes, not as evidence for or against any method.
