"""On-demand ONNX export of a published baseline at one padded shape, with gate P1 parity.

Runs in the training environment (torch 2.2.2). Official code lives in
<LACUNAFRAME>/.baselines (pinned commits, see baselines-manifest.json) and
weights in $LACUNA_EXTERNAL/weights/src. Parity is checked on real anchor
pairs supplied by the caller (an .npz with `pairs`, shape (K, 2, H, W, 3) uint8).

    <train-venv>/bin/python research/metrics/export_shape.py <model> <pairs.npz>
"""
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[2]
EXTERNAL = Path(os.environ.get('LACUNA_EXTERNAL', REPO.parent / 'lacuna-data')).resolve()
SRC, CODE, OUT = EXTERNAL / 'weights' / 'src', REPO.parent / '.baselines', EXTERNAL / 'weights'
torch.set_grad_enabled(False)
SPEC = {  # model: (repo, divisor, padding)
    'amt_s': ('AMT', 16, 'replicate_center'), 'amt_s_gopro': ('AMT', 16, 'replicate_center'),
    'ifrnet_s': ('IFRNet', 16, 'replicate_center'), 'ifrnet': ('IFRNet', 16, 'replicate_center'),
    'ifrnet_s_gopro': ('IFRNet', 16, 'replicate_center'), 'ifrnet_gopro': ('IFRNet', 16, 'replicate_center'),
    'rife_paper': ('ECCV2022-RIFE', 32, 'zero_bottom_right'),
}


def build(name: str) -> torch.nn.Module:
    repo = SPEC[name][0]
    sys.path.insert(0, str(CODE / repo))
    if repo == 'AMT':
        Model = importlib.import_module('networks.AMT-S').Model
        m = Model(corr_radius=3, corr_lvls=4, num_flows=3)
        m.load_state_dict(torch.load(SRC / ('gopro_amt-s.pth' if name.endswith('gopro') else 'amt-s.pth'), map_location='cpu')['state_dict'])
        fwd = lambda a, b, e: m(a, b, e, scale_factor=1.0, eval=True)['imgt_pred']
    elif repo == 'IFRNet':
        small = name.startswith('ifrnet_s')
        Model = importlib.import_module('models.IFRNet_S' if small else 'models.IFRNet').Model
        m = Model()
        file = f"IFRNet{'_S' if small else ''}_{'GoPro' if name.endswith('gopro') else 'Vimeo90K'}.pth"
        m.load_state_dict(torch.load(SRC / file, map_location='cpu'))
        fwd = lambda a, b, e: m.inference(a, b, e)
    else:
        from model.IFNet import IFNet
        m = IFNet()
        state = torch.load(SRC / 'flownet.pkl', map_location='cpu')
        m.load_state_dict({k.replace('module.', ''): v for k, v in state.items() if 'module.' in k})
        fwd = lambda a, b, e: m(torch.cat((a, b), 1), scale=[4, 2, 1])[2][2]
    m.eval()

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = m

        def forward(self, img0, img1, embt):
            return fwd(img0, img1, embt)
    return Wrap()


def pad(x: torch.Tensor, divisor: int, mode: str) -> torch.Tensor:
    h, w = x.shape[-2:]
    ph, pw = (-h) % divisor, (-w) % divisor
    if mode == 'replicate_center':
        return torch.nn.functional.pad(x, [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2], mode='replicate')
    return torch.nn.functional.pad(x, [0, pw, 0, ph])


def main() -> None:
    name, pairs = sys.argv[1], np.load(sys.argv[2])['pairs']
    _, divisor, mode = SPEC[name]
    module = build(name)
    t = lambda a: pad(torch.from_numpy(a.astype(np.float32).transpose(2, 0, 1)[None] / 255.), divisor, mode)
    x0, x1 = t(pairs[0, 0]), t(pairs[0, 1])
    shape = f'{x0.shape[2]}x{x0.shape[3]}'
    path = OUT / f'{name}_{shape}.onnx'
    torch.onnx.export(module, (x0, x1, torch.full((1, 1, 1, 1), .5)), str(path), input_names=['img0', 'img1', 'embt'], output_names=['imgt'], opset_version=17)
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    errors = []
    for a, b in pairs:
        y0, y1, e = t(a), t(b), torch.full((1, 1, 1, 1), .5)
        ref = module(y0, y1, e).numpy()
        feeds = {'img0': y0.numpy(), 'img1': y1.numpy(), 'embt': e.numpy()}
        got = session.run(None, {i.name: feeds[i.name] for i in session.get_inputs()})[0]
        errors.append(float(np.abs(ref - got).max()))
    manifest_path = OUT / 'baselines-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    entry = manifest.setdefault(name, dict(divisor=divisor, padding=mode, files={}))
    entry['files'][shape] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
                                 p1_max_abs_error=max(errors), p1_pairs=len(pairs), p1_passed=max(errors) < 1e-3)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({name: entry['files'][shape]}))


if __name__ == '__main__':
    main()
