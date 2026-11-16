---
license: mit
library_name: onnx
pipeline_tag: video-to-video
tags:
  - rife
  - frame-interpolation
  - video
  - onnx
  - webgpu
---

# RIFE v4.9 (ONNX)

`rife49.onnx` — an ONNX export of **RIFE v4.9** for video frame interpolation
(synthesising the in-between frames when raising a video's frame rate). Runs
client-side with onnxruntime-web on the WASM or WebGPU execution providers.

## Inputs / outputs

- **Inputs:** `img0` float32 `[1, 3, H, W]`, `img1` float32 `[1, 3, H, W]`,
  `timestep` float32 `[1]`
- **Output:** `output` float32 `[1, 3, H, W]` — the interpolated frame at
  `timestep` (0..1) between `img0` and `img1`
- Planar RGB (CHW), values in 0..1. `H` and `W` are padded to a multiple of 32
  for reference quality.

## Provenance & license

- **Algorithm:** RIFE — *Real-Time Intermediate Flow Estimation for Video Frame
  Interpolation* (Megvii, ECCV 2022), original code MIT
  (`Copyright (c) Megvii Inc.`).
- **Weights:** [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE)
  (v4.9). Practical-RIFE releases both its code **and its model weights** under
  the **MIT license** — its README states the model links are "under the same
  MIT license as this project."
- **ONNX export:**
  [yuvraj108c/rife-onnx](https://huggingface.co/yuvraj108c/rife-onnx)
  (`rife49_ensemble_True_scale_1_sim.onnx`).

**License: MIT.** See [`LICENSE`](./RIFE-LICENSE) (`Copyright (c) 2021 hzwer`). This
repository mirrors the model so it can be fetched at runtime by the Edge Tools
app; a format conversion of MIT-licensed weights remains MIT.
