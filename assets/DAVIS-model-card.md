---
license: cc-by-nc-4.0
task_categories:
  - video-classification
  - object-detection
  - image-segmentation
tags:
  - davis
  - davis-2017
  - video-object-segmentation
  - mp4
  - computer-vision
  - tracking
size_categories:
  - n<1K
---

# DAVIS 2017 — 480p MP4 Sequences

Pre-encoded MP4 versions of all **90 sequences** from the
[DAVIS 2017 trainval split](https://davischallenge.org/) at **480p resolution**.

Useful if you want to quickly load, stream, or embed DAVIS sequences without
running `ffmpeg` yourself or dealing with per-frame JPEG folders.

## Contents

| File pattern | Description |
|---|---|
| `<seq>_raw_24fps.mp4` | Raw RGB frames, no annotation, 24 fps |
| `<seq>_ov055_24fps.mp4` | RGB + DAVIS palette mask overlay at **55% opacity**, 24 fps |
| `<seq>_ov010_24fps.mp4` | RGB + DAVIS palette mask overlay at **10% opacity** (subtle), 24 fps |

- **90 sequences** × up to 3 variants = 185 MP4 files  
- Total size: ~290 MB  
- Codec: H.264 (`libx264`, `yuv420p`, CRF 18, `faststart`)  
- Source annotations: per-pixel palette-indexed PNGs where pixel value = object ID

## Usage

```python
from huggingface_hub import hf_hub_download

# Download one sequence (raw)
path = hf_hub_download(
    repo_id="emirkisa/DAVIS-2017-480p-mp4",
    filename="camel_raw_24fps.mp4",
    repo_type="dataset",
)

# Download everything (~290 MB)
from huggingface_hub import snapshot_download
local_dir = snapshot_download(
    repo_id="emirkisa/DAVIS-2017-480p-mp4",
    repo_type="dataset",
)
```

## DAVIS Colour Palette (overlay videos)

Objects are coloured with the official DAVIS 20-colour palette:  
object 1 → `#800000`, object 2 → `#008000`, object 3 → `#808000`, …  
Background pixels are fully transparent (not blended).

## Splits

The 90 sequences span both DAVIS-2016 and DAVIS-2017:

| Split | Sequences |
|---|---|
| DAVIS-2016 train | 30 |
| DAVIS-2016 val | 20 |
| DAVIS-2017 train | 60 |
| DAVIS-2017 val | 30 |

(Sequences appear in multiple splits; the total unique count is 90.)

## License

Inherited from the original DAVIS dataset:  
**[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)** — free for non-commercial use with attribution.

## Citation

```bibtex
@article{Pont-Tuset_arXiv_2017,
  author  = {Jordi Pont-Tuset and Federico Perazzi and Sergi Caelles and
             Pablo Arbelàez and Alexander Sorkine-Hornung and Luc Van Gool},
  title   = {The 2017 {DAVIS} Challenge on Video Object Segmentation},
  journal = {arXiv:1704.00675},
  year    = {2017}
}
```
