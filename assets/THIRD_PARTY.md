# External assets — not covered by the repository's MIT license

## Pretrained RIFE

`models/rife49.onnx`: RIFE 4.9 ensemble, FP32, exported by yuvraj108c from hzwer/Practical-RIFE. Upstream code and model weights are MIT-licensed; see `RIFE-LICENSE`. The exact source revision and SHA-256 are in `manifest.json`. `RIFE-model-card.md` preserves a separate mirror's description of the same exported artifact, including RGB layout and padding conventions. We use the pinned yuvraj108c artifact, not an unversioned automatic download.

Primary algorithm: Huang et al., Real-Time Intermediate Flow Estimation for Video Frame Interpolation, ECCV 2022, https://arxiv.org/abs/2011.06294. Practical checkpoints may differ from the original paper's evaluated architecture. No affiliation or endorsement is implied.

## Natural DAVIS footage

`video/blackswan.mp4`, `video/bmx-trees.mp4`, `video/breakdance.mp4` are downloaded from emirkisa's **raw (no annotation overlay)** MP4 conversion of DAVIS 2017 trainval, pinned at revision `872ccd3c5d4d9c98016798e7af0342b33ec6ab4d`.

Mirror: https://huggingface.co/datasets/emirkisa/DAVIS-2017-480p-mp4
Original dataset: https://davischallenge.org/
Dataset attribution: Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, Alexander Sorkine-Hornung, “A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation,” CVPR 2016; Jordi Pont-Tuset et al., “The 2017 DAVIS Challenge on Video Object Segmentation,” 2017. MP4 conversion: emirkisa.

The mirror declares **Creative Commons Attribution–NonCommercial 4.0**, https://creativecommons.org/licenses/by-nc/4.0/. We retain its dataset card in `DAVIS-model-card.md`. These clips and derived natural-video exports/screenshots are for attributed noncommercial research/education; the MIT license on our code does not relicense them. For commercial redistribution, replace the footage or obtain the necessary rights.

Changes: backend letterboxes to 320×180, samples every second frame from the mirror's 24fps sequence, caps at 48 samples, and applies documented corruptions/restorations. Original downloaded files are retained unchanged. The mirror's “raw” label means no mask overlay, **not uncompressed camera RAW**. Evaluation uses decoded compressed frames, not pristine sensor data. Manifest hashes identify the exact source files.

## Optional RIFE 4.26 candidate

`models/rife426.onnx`: Practical-RIFE 4.26 FP32, vs-mlrt ONNX export mirrored by kortique/photoview-ai-models. Pinned revision, URL and SHA-256 are in `candidate.json`. Upstream RIFE MIT terms apply; see `RIFE-LICENSE`. This separate research experiment is not the UI default. The adapter follows vs-mlrt's documented coordinate ABI, with independently written NumPy packing. Model initialization verifies the artifact checksum.

## Bundled interface fonts

Instrument Sans, Instrument Serif and IBM Plex Mono are distributed through @fontsource under the SIL Open Font License. Copies of their license notices are in `font-licenses/`.

Documentation note: the preserved RIFE mirror card’s relative `LICENSE` link is adjusted to `RIFE-LICENSE` in this repository; no model metadata or license terms are changed.
