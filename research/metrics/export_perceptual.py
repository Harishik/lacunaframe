"""One-off export of LPIPS (AlexNet, v0.1) and DISTS to ONNX for CPU evaluation.

Runs in a temporary PyTorch environment (torch 2.2.2, numpy<2, lpips, DISTS_pytorch);
the main project never imports torch. Each export is checked against the
PyTorch reference on random and structured inputs before it is kept.

    TORCH_HOME=$TMPDIR/torch-home <export-venv>/bin/python research/metrics/export_perceptual.py <out_dir>
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)


class LpipsWrap(torch.nn.Module):
    """Inputs: RGB float32 NCHW in [0, 1]. Output: per-image LPIPS distance (N,)."""
    def __init__(self):
        super().__init__()
        import lpips
        self.m = lpips.LPIPS(net='alex', version='0.1', verbose=False).eval()

    def forward(self, a, b):
        return self.m(a * 2 - 1, b * 2 - 1).flatten()


class DistsWrap(torch.nn.Module):
    """Inputs: RGB float32 NCHW in [0, 1]. Output: per-image DISTS distance (N,)."""
    def __init__(self):
        super().__init__()
        from DISTS_pytorch import DISTS
        self.m = DISTS().eval()

    def forward(self, a, b):
        return self.m(a, b, require_grad=False, batch_average=False).flatten()


def export(name, module):
    import onnxruntime  # noqa: F401  (validation below uses ORT if available)
    path = out / f'{name}.onnx'
    a, b = torch.rand(1, 3, 64, 96), torch.rand(1, 3, 64, 96)
    torch.onnx.export(module, (a, b), str(path), input_names=['a', 'b'], output_names=['distance'], opset_version=17,
                      dynamic_axes={'a': {0: 'n', 2: 'h', 3: 'w'}, 'b': {0: 'n', 2: 'h', 3: 'w'}, 'distance': {0: 'n'}})
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    # Validate against a float64 copy of the reference model (a float32 eager
    # reference proved an unreliable yardstick during development).
    import copy
    reference = copy.deepcopy(module).double().eval()
    errors = []
    for shape in [(2, 3, 90, 160), (1, 3, 180, 320), (1, 3, 256, 448)]:
        x = torch.rand(*shape, dtype=torch.float64)
        y = (x + 0.1 * torch.randn(*shape, dtype=torch.float64)).clamp(0, 1)
        with torch.no_grad():
            ref = reference(x, y).numpy()
        got = s.run(None, {'a': x.float().numpy(), 'b': y.float().numpy()})[0]
        errors.append(float(np.max(np.abs(ref - got))))
    assert max(errors) < 1e-4, (name, errors)
    return dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
                max_abs_error_vs_float64_torch=max(errors), input='RGB float32 NCHW in [0,1]', output='per-image distance')


torch.set_grad_enabled(False)
if True:
    manifest = dict(lpips_alex=export('lpips_alex_v0.1', LpipsWrap()), dists=export('dists', DistsWrap()),
                    torch=torch.__version__)
(out / 'perceptual-manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
