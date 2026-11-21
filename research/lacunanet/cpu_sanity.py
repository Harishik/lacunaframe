"""Learning-signal check: v1 vs v2 under an identical small CPU budget (dev frames only).

Both models see the same GOPRO-train samples (same seed), the same steps, batch and crop.
After training, each reconstructs held-out dev missing frames (SNU YouTube, gaps 1-8) and is
compared with temporal linear blending. Design check only; no test data.

    ../.venv-train/bin/python research/lacunanet/cpu_sanity.py [steps]
"""
import json
import os
import random
import sys
import time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from research.lacunanet.data import GoproSource
from research.lacunanet.model import LacunaNet, make_cond
from research.lacunanet.model_v2 import LacunaNetV2, anchor_loss_multiscale
from research.lacunabench.recoverability import brackets
from research.scripts.lacunabench_v1 import tasks

EXT = REPO.parent / 'lacuna-data'
BATCH, CROP_SEED = 4, 7
LPIPS_W = float(os.environ.get('SANITY_LPIPS', '0.1'))


def train(model, steps, src, v2):
    import lpips
    per = lpips.LPIPS(net='alex', verbose=False).eval()
    for p in per.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = random.Random(CROP_SEED)
    t0, log = time.time(), []
    for step in range(steps):
        crops, conds = [], []
        for _ in range(BATCH):
            c, cond, (m2, p2) = src.sample(rng)
            if not m2: c[2] = np.zeros_like(c[2])
            if not p2: c[3] = np.zeros_like(c[3])
            crops.append(np.stack([np.ascontiguousarray(x) for x in c])); conds.append(cond)
        x = torch.from_numpy(np.stack(crops)).permute(1, 0, 4, 2, 3).float() / 255.
        cond = torch.tensor(conds, dtype=torch.float32)
        if v2:
            frame, scale, aux = model(x[0], x[1], x[2], x[3], cond, return_aux=True)
        else:
            frame, scale = model(x[0], x[1], x[2], x[3], cond)
        rec = torch.sqrt((frame - x[4]) ** 2 + 1e-6).mean()
        loss = rec + LPIPS_W * per(frame * 2 - 1, x[4] * 2 - 1).mean() if LPIPS_W else rec
        resid = (frame.detach() - x[4]).abs().mean(1, keepdim=True)
        loss = loss + 0.01 * (resid / scale + torch.log(scale)).mean()
        if v2:
            loss = loss + 0.5 * anchor_loss_multiscale(x[0], x[1], aux['pyramid'])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (step + 1) % 50 == 0:
            log.append(dict(step=step + 1, rec=round(rec.item(), 4), sec=round(time.time() - t0)))
            print(type(model).__name__, log[-1], flush=True)
    return log


@torch.no_grad()
def evaluate(model):
    model.eval()
    splits = json.loads((REPO / 'research/protocols/splits-v1.json').read_text())
    t = lambda a: torch.from_numpy(a.astype(np.float32).transpose(2, 0, 1)[None] / 255.)
    res = []
    for c in splits['dev'][:int(__import__('os').environ.get('SANITY_CLIPS', '3'))]:
        frames = np.load(EXT / 'snu-film' / c['file'])['frames'][:, :352]      # crop to a multiple of 16
        for task in tasks(c['id'], len(frames)):
            if task['task'] not in ('gap1', 'gap2', 'gap4', 'gap8'):
                continue
            for left, right, l2, r2, idx in brackets(task['missing']):
                span, gap = right - left, right - left - 1
                for i in idx:
                    tt = (i - left) / span
                    cond = torch.tensor([make_cond(tt, gap, -tt - (left - l2) / span if l2 is not None else 0,
                                                   1 - tt + (r2 - right) / span if r2 is not None else 0, l2 is not None, r2 is not None)])
                    z = torch.zeros_like(t(frames[left]))
                    args = (t(frames[left]), t(frames[right]), t(frames[l2]) if l2 is not None else z, t(frames[r2]) if r2 is not None else z, cond)
                    if isinstance(model, LacunaNetV2):
                        f, _, aux = model(*args, return_aux=True)
                        base = aux['base'][0].numpy().transpose(1, 2, 0) * 255
                    else:
                        f, _ = model(*args); base = None
                    est = f[0].numpy().transpose(1, 2, 0) * 255
                    lin = (1 - tt) * frames[left].astype(float) + tt * frames[right]
                    ps = lambda e: 10 * np.log10(255 ** 2 / max(np.mean((np.clip(e, 0, 255).round() - frames[i]) ** 2), 1e-9))
                    res.append(dict(task=task['task'], model=ps(est), linear=ps(lin), base=None if base is None else ps(base)))
    model.train()
    out = {}
    for task in ('gap1', 'gap2', 'gap4', 'gap8'):
        sub = [r for r in res if r['task'] == task]
        out[task] = dict(n=len(sub), model=round(float(np.mean([r['model'] for r in sub])), 2), linear=round(float(np.mean([r['linear'] for r in sub])), 2))
        if sub and sub[0]['base'] is not None:
            out[task]['warp_blend_only'] = round(float(np.mean([r['base'] for r in sub])), 2)
    return out


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    torch.set_num_threads(6)
    src = GoproSource(EXT / 'train' / 'gopro')
    report = dict(steps=steps, batch=BATCH, gopro_scenes=len(src.scenes))
    which = __import__('os').environ.get('SANITY_MODELS', 'v1,v2').split(',')
    models = (('v1_C32', lambda: LacunaNet(32), False), ('v2_C24', lambda: LacunaNetV2(24), True), ('v3_C24', lambda: LacunaNetV2(24, prior='lk'), True))
    for name, ctor, v2 in [m for m in models if m[0][:2] in which]:
        torch.manual_seed(0)
        model = ctor()
        if os.environ.get('SANITY_INIT_EVAL'):
            report[name + '_init'] = dict(dev=evaluate(model)); print(name, 'init', report[name + '_init']['dev'], flush=True)
        log = train(model, steps, src, v2)
        report[name] = dict(train=log, dev=evaluate(model))
        print(name, report[name]['dev'], flush=True)
    report['lpips_weight'] = LPIPS_W
    path = REPO / f"research/results/lacunanet/cpu-sanity-{'-'.join(which)}-{steps}{'' if LPIPS_W == 0.1 else f'-lpips{LPIPS_W:g}'}.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v['dev'] for k, v in report.items() if isinstance(v, dict) and 'dev' in v}, indent=2))


if __name__ == '__main__':
    main()
