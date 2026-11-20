# Cross-platform portability findings (2026-11-20)

Rerunning the archived evidence on a second host found two problems, which have been fixed. Neither changes an archived number.

## 1. Reference frames depended on the OS video decoder (fixed)

`opencv-python-headless` decodes through FFmpeg on Linux and through AVFoundation on macOS. On DAVIS `bmx-trees` frame 0, the two backends differ by a **mean of 3.0 and a maximum of 71** 8-bit levels. Every reference frame, and therefore every PSNR/SSIM, silently depended on which OS ran the benchmark. On macOS the archived scores could not be reproduced: one rescored triplet was off by 0.23 dB.

**Fix:** [`backend/decode.py`](../../../backend/decode.py) decodes everything through the pinned `imageio-ffmpeg` binary. App sources, uploads, the research pilot, the profiler, the documentation builder and artifact verification all use it. FFmpeg decoding reproduces the archived Linux scores **bit-exactly** (Δ = 0.0 dB on all rescored triplets), and all 120 archived MP4s decode to their recorded frame counts.

## 2. Static INT8 outputs are runtime-dependent (documented, test made explicit)

The same quantized ONNX files were run on the archived host (Linux, ORT 1.31.1) and on an Intel MacBook (macOS 26.7.1, ORT 1.23.2, the newest ORT with an Intel-macOS wheel):

| Variant | Mean abs diff | Max abs diff | PSNR vs archive |
|---|---:|---:|---:|
| FP32 4.9 | 0.0001 | 1 | 88.5 dB |
| Dynamic U8 | 0 | 0 | identical |
| QDQ trunk | 0 | 0 | identical |
| **QDQ all-Conv** | **2.03** | **108** | **35.1 dB** |

Static all-Conv QDQ is the only graph whose output changes materially across runtime and CPU. A quantized model's quality is therefore a property of *(graph, runtime version, CPU kernels)*, not of the graph alone. Any compression claim must report all three, and must be re-measured on each deployment target.

The evidence test now requires an exact match only when the runtime and OS family equal the archived environment. Otherwise it records the drift here (`drift-*.json`) and fails only below 30 dB. Raw record: [drift-ort1.23.2-darwin-x86_64.json](drift-ort1.23.2-darwin-x86_64.json).

Environment for this host: `backend/requirements-lock-macos-x86_64.txt`. The original Linux lock is unchanged.
