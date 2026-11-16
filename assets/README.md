# Pinned inputs and rights

[Repository map](../docs/guides/REPOSITORY.md) · [Full provenance](THIRD_PARTY.md)

| Path | Contents | Pin / rights |
|---|---|---|
| `video/` | Black swan, BMX-trees and breakdance recorded sequences | Immutable mirror revision and SHA-256 in `manifest.json`; CC BY-NC 4.0 |
| `models/rife49.onnx` | Default RIFE 4.9 ensemble FP32 export | Revision/hash in `manifest.json`; upstream MIT |
| `models/rife426.onnx` | Optional RIFE 4.26 FP32 candidate | Revision/hash in `candidate.json`; upstream MIT |
| `font-licenses/` | Bundled interface typography notices | SIL Open Font License |
| `*-model-card.md` | Preserved mirror descriptions | Attribution/context, not a substitute for artifact hashes |

Both model binaries and all three original clips are included in Git. Run `.venv/bin/python scripts/fetch_assets.py --candidate` from the repository root to verify every hash or explicitly fetch missing/corrupt copies. Inference makes no network request. These are pretrained inference artifacts; no new weights were trained or distilled for this project.
