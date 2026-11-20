"""One-off ONNX export of published interpolators with an export-parity check (protocol v2, gate P1).

Run inside a temporary PyTorch environment that has the official repositories
cloned next to their weights (see research/results/baselines/README.md):

    <venv>/bin/python research/metrics/export_baselines.py <work_dir> <out_dir> <snu_dev_npz> <vimeo_npz> <model>

Run once per model (amt_s, ifrnet_s, ifrnet, rife_paper): the official repositories
share top-level module names (e.g. `utils`), so each export needs a fresh process.

Graphs are exported at fixed shapes (the official code builds sampling grids
from tensor sizes), one per padded resolution: 368x640 for LacunaBench frames
(640x360, padded to a multiple of 16) and 256x448 for Vimeo90K. Inputs are
already padded by the caller exactly as the official inference code pads them.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort

work, out = Path(sys.argv[1]), Path(sys.argv[2])
snu, vimeo = np.load(sys.argv[3])['frames'], np.load(sys.argv[4])['triplets']
out.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)


def pad_replicate_center(x: torch.Tensor, divisor: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    ph, pw = (-h) % divisor, (-w) % divisor
    return torch.nn.functional.pad(x, [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2], mode='replicate')


class AMTWrap(torch.nn.Module):
    def __init__(self, checkpoint: str = 'amt-s.pth'):
        super().__init__()
        sys.path.insert(0, str(work / 'AMT'))
        import importlib
        Model = importlib.import_module('networks.AMT-S').Model
        self.m = Model(corr_radius=3, corr_lvls=4, num_flows=3)
        self.m.load_state_dict(torch.load(work / checkpoint, map_location='cpu')['state_dict'])
        self.m.eval()

    def forward(self, img0, img1, embt):
        return self.m(img0, img1, embt, scale_factor=1.0, eval=True)['imgt_pred']


class IFRNetWrap(torch.nn.Module):
    def __init__(self, small: bool, dataset: str = 'Vimeo90K'):
        super().__init__()
        sys.path.insert(0, str(work / 'IFRNet'))
        import importlib
        Model = importlib.import_module('models.IFRNet_S' if small else 'models.IFRNet').Model
        self.m = Model()
        name = f'IFRNet_small/IFRNet_S_{dataset}.pth' if small else f'IFRNet/IFRNet_{dataset}.pth'
        self.m.load_state_dict(torch.load(work / 'ifrnet/checkpoints' / name, map_location='cpu'))
        self.m.eval()

    def forward(self, img0, img1, embt):
        return self.m.inference(img0, img1, embt)


class RIFEPaperWrap(torch.nn.Module):
    """ECCV 2022 paper model. Its IFNet ignores `timestep`, so it is midpoint-only; embt is accepted and unused."""
    def __init__(self):
        super().__init__()
        sys.path.insert(0, str(work / 'ECCV2022-RIFE'))
        from model.IFNet import IFNet
        self.m = IFNet()
        state = torch.load(work / 'rife_paper/train_log/flownet.pkl', map_location='cpu')
        self.m.load_state_dict({k.replace('module.', ''): v for k, v in state.items() if 'module.' in k})
        self.m.eval()

    def forward(self, img0, img1, embt):
        merged = self.m(torch.cat((img0, img1), 1), scale=[4, 2, 1])[2]
        return merged[2]


def tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a.astype(np.float32).transpose(2, 0, 1)[None] / 255.)


def export(name: str, module: torch.nn.Module, divisor: int, pad: str, shapes: dict) -> dict:
    entry = dict(divisor=divisor, padding=pad, files={})
    for tag, pairs in shapes.items():
        a0, b0 = pairs[0]
        x0, x1 = tensor(a0), tensor(b0)
        if pad == 'replicate_center':
            x0, x1 = pad_replicate_center(x0, divisor), pad_replicate_center(x1, divisor)
        h, w = x0.shape[-2:]
        path = out / f'{name}_{h}x{w}.onnx'
        embt = torch.full((1, 1, 1, 1), .5)
        torch.onnx.export(module, (x0, x1, embt), str(path), input_names=['img0', 'img1', 'embt'], output_names=['imgt'], opset_version=17)
        session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        errors = []
        for a, b in pairs:
            y0, y1 = tensor(a), tensor(b)
            if pad == 'replicate_center':
                y0, y1 = pad_replicate_center(y0, divisor), pad_replicate_center(y1, divisor)
            for t in ([.5] if name == 'rife_paper' else [.125, .25, .5]):
                e = torch.full((1, 1, 1, 1), t)
                ref = module(y0, y1, e).numpy()
                feeds = {'img0': y0.numpy(), 'img1': y1.numpy(), 'embt': e.numpy()}
                got = session.run(None, {i.name: feeds[i.name] for i in session.get_inputs()})[0]  # unused inputs are pruned
                errors.append(float(np.abs(ref - got).max()))
        entry['files'][f'{h}x{w}'] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
                                          p1_max_abs_error=max(errors), p1_pairs=len(pairs), p1_passed=max(errors) < 1e-3)
        print(name, tag, entry['files'][f'{h}x{w}'], flush=True)
    return entry


snu_pairs = [(snu[i], snu[i + 4]) for i in range(0, 32, 4)]          # 8 real pairs, 640x360
vimeo_pairs = [(t[0], t[2]) for t in vimeo[::32][:8]]                 # 8 real Vimeo pairs, 448x256
which = sys.argv[5]
path = out / 'baselines-manifest.json'
manifest = json.loads(path.read_text()) if path.exists() else {}
both = {'lacuna': snu_pairs, 'vimeo': vimeo_pairs}
if which == 'amt_s':
    manifest['amt_s'] = export('amt_s', AMTWrap(), 16, 'replicate_center', both)
elif which in ('ifrnet_s', 'ifrnet'):
    manifest[which] = export(which, IFRNetWrap(which == 'ifrnet_s'), 16, 'replicate_center', both)
elif which == 'amt_s_gopro':
    manifest[which] = export(which, AMTWrap('gopro_amt-s.pth'), 16, 'replicate_center', {'lacuna': snu_pairs})
elif which in ('ifrnet_s_gopro', 'ifrnet_gopro'):
    manifest[which] = export(which, IFRNetWrap(which == 'ifrnet_s_gopro', 'GoPro'), 16, 'replicate_center', {'lacuna': snu_pairs})
elif which == 'rife_paper':
    manifest['rife_paper'] = export('rife_paper', RIFEPaperWrap(), 32, 'zero_bottom_right', {'vimeo': vimeo_pairs})
    manifest['rife_paper']['note'] = 'Midpoint-only: the ECCV 2022 IFNet ignores its timestep argument. Used for Vimeo90K parity (P2) only.'
else:
    raise SystemExit(f'unknown model {which}')
manifest['torch'] = torch.__version__
path.write_text(json.dumps(manifest, indent=2))
