# Multi-anchor flow: diagnostic check

This is **not** the pre-registered G1 gate: DAVIS is diagnostic-only under protocol v1, and G1 needs the frozen test split. Unit of analysis: scene, with seeds averaged within scene.

Parity check: two-anchor flow reproduces the archived natural-v02 scores (max |Δ| = 0.00e+00 dB).

| Condition | Flow (dB) | Quadratic (dB) | Δ (scene mean) | 95% CI of Δ | Scenes won | s / missing frame (flow → quad) |
|---|---:|---:|---:|---|---:|---|
| scattered | 19.513 | 19.647 | +0.134 | [+0.016, +0.254] | 3/3 | 0.069 → 0.121 |
| long-gap | 17.605 | 17.709 | +0.104 | [-0.185, +0.493] | 2/3 | 0.072 → 0.075 |
| synthetic-long-gap | 26.568 | 26.563 | -0.004 | [-0.266, +0.138] | 2/3 | 0.071 → 0.085 |

With three scenes per group these intervals describe these clips only. The decision on multi-anchor reconstruction waits for G1 on the test split.
