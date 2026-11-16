# Lossless external-comparison exchange / v0.3

[Archive map](../README.md) · [Full exchange contract](../../docs/guides/EXPERIMENTS.md#4-compare-an-outside-model-on-identical-pixels)

BMX natural footage, long-gap regime, seed 42. This is a completed export → actual linear restoration → timestamp-checked evaluation path. It demonstrates the exchange contract; it is not a result from an outside publication's model.

| File | Purpose |
|---|---|
| `inputs.npz` | Damaged RGB uint8 NHWC frames, missing flags, spatial masks, exact timestamps and FPS; solver-readable |
| `targets.npz` | Untouched reference and affected support; evaluator-only |
| `predictions.npz` | Actual linear-baseline output, aligned with exact copied timestamps |
| `protocol.json` | Source/configuration/layout and target-access rules |
| `result.json` | Genuine evaluation and SHA-256 of target/prediction artifacts |

Never give `targets.npz` to a restoration model. The file exists here for independent scoring and reproducibility. Arrays are loaded with `allow_pickle=False`. The source is compressed DAVIS mirror media normalized to 320×180, not sensor RAW.

The documentation generator uses these exact observations and targets to execute linear, flow and RIFE restoration and derive its comparison figures. CC BY-NC 4.0 applies to the natural-media arrays and derivatives.
