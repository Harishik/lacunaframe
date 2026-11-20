# Data ledger (protocol v1)

Storage cap: **6 GiB** for all external data and weights. Stored now: **0.81 GB** in data arrays, plus 0.07 GB of perceptual-metric ONNX weights. Data lives outside git in `$LACUNA_EXTERNAL` (default `../lacuna-data`). Machine-readable: [ledger.json](ledger.json).

| Source | Pinned revision | Selection | Fetched (transient) | Stored |
|---|---|---|---:|---:|
| [snu-film](https://huggingface.co/datasets/xypeng9903/SNU_FILM) | `a02ce2ca38` | 48-frame centred window at stride 8 from each of 31 test sequences; centre-crop 16:9, INTER_AREA to 640x360 | 1421 MB | 635 MB |
| [vimeo-triplets](https://huggingface.co/datasets/paschalc24/vimeo-90k-test-full) | `7af0ffe8b6` | 256 evenly spaced triplets from data/test-00000-of-00004.parquet; native resolution; shard deleted after extraction | 448 MB | 172 MB |
| Perceptual weights | exported locally from `lpips` 0.1 (AlexNet) and `DISTS_pytorch` with torch 2.2.2 | LPIPS 9.9 MB, DISTS 58.9 MB ONNX, validated against float64 PyTorch | ≈0.8 GB (torchvision AlexNet + VGG16, deleted) | 69 MB |

Licences: SNU-FILM (Choi et al., AAAI 2020); its GoPro clips derive from GOPRO (Nah et al., CVPR 2017, CC BY 4.0), and its YouTube clips are for research use. Vimeo90K (Xue et al., IJCV 2019) is for research use. Derived arrays are kept locally and never redistributed in this repository.

Reproduce: `.venv/bin/python research/data/fetch.py snu-film` and `... vimeo-triplets`. The fetcher refuses to exceed the cap and deletes originals after conversion.
