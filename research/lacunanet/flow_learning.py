"""Does the pair-flow estimator learn motion? Encoder + PairFlow trained alone with the multiscale
anchor loss on GOPRO-train crops; end-point error (full-resolution px) against Farneback flow on
dev anchor pairs, at pixels where Farneback is forward/backward consistent. Dev only.

    ../.venv-train/bin/python research/lacunanet/flow_learning.py <steps> <variant[,variant]>
    variants: learned (v2 as designed), lk (v3: Lucas-Kanade prior + learned residual)
"""
import json, random, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from research.lacunanet.data import GoproSource
from research.lacunanet.model_v2 import Encoder4, PairFlow, anchor_loss_multiscale, warp

EXT = REPO.parent / 'lacuna-data'
FARNEBACK = (.5, 3, 21, 4, 7, 1.5, 0)


def dev_pairs(n_clips=6, gaps=(1, 4)):
    splits = json.loads((REPO / 'research/protocols/splits-v1.json').read_text())
    pairs = []
    for c in splits['dev'][:n_clips]:
        fr = np.load(EXT / 'snu-film' / c['file'])['frames'][:, :352]
        for g in gaps:
            for s in (5, len(fr) // 2):
                a, b = fr[s], fr[s + g + 1]
                ga, gb = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
                fab = cv2.calcOpticalFlowFarneback(ga, gb, None, *FARNEBACK)
                fba = cv2.calcOpticalFlowFarneback(gb, ga, None, *FARNEBACK)
                h, w = ga.shape
                gx, gy = np.meshgrid(np.arange(w), np.arange(h))
                back = cv2.remap(fba, (gx + fab[..., 0]).astype(np.float32), (gy + fab[..., 1]).astype(np.float32), cv2.INTER_LINEAR)
                ok = np.linalg.norm(fab + back, axis=2) < 1.0
                pairs.append(dict(gap=g, a=a, b=b, ref=fab, ok=ok))
    return pairs


class Net(nn.Module):
    def __init__(self, c, hybrid):
        super().__init__()
        self.enc, self.pair = Encoder4(c), PairFlow(c, 'lk' if hybrid else None)

    def forward(self, a, b, all_levels=True):
        ea, eb = self.enc(a), self.enc(b)
        return self.pair(ea, eb, all_levels=all_levels, imgs=(a, b)), self.pair(eb, ea, all_levels=all_levels, imgs=(b, a))


T = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32).transpose(2, 0, 1)[None] / 255.)


@torch.no_grad()
def epe(net, pairs):
    net.eval(); out = {}
    for g in sorted({p['gap'] for p in pairs}):
        errs, zero = [], []
        for p in [p for p in pairs if p['gap'] == g]:
            (f01, _), _ = net(T(p['a']), T(p['b'])), None
            f = f01[-1]
            f = F.interpolate(f, size=p['a'].shape[:2], mode='bilinear', align_corners=False) * 4
            f = f[0].numpy().transpose(1, 2, 0)
            errs.append(np.linalg.norm(f - p['ref'], axis=2)[p['ok']].mean())
            zero.append(np.linalg.norm(p['ref'], axis=2)[p['ok']].mean())
        out[f'gap{g}'] = dict(epe=round(float(np.mean(errs)), 2), zero_flow=round(float(np.mean(zero)), 2))
    net.train(); return out


def main():
    steps = int(sys.argv[1]); variants = sys.argv[2].split(',')
    torch.set_num_threads(6)
    src = GoproSource(EXT / 'train' / 'gopro'); pairs = dev_pairs()
    report = {}
    for v in variants:
        torch.manual_seed(0); rng = random.Random(7)
        net = Net(24, v == 'lk')
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
        curve = [dict(step=0, **epe(net, pairs))]; print(v, curve[-1], flush=True); t0 = time.time()
        for step in range(steps):
            xs = []
            for _ in range(4):
                c, _, _ = src.sample(rng)
                xs.append(np.stack([c[0], c[1]]))
            x = torch.from_numpy(np.stack(xs)).permute(1, 0, 4, 2, 3).float() / 255.
            p01, p10 = net(x[0], x[1])
            loss = anchor_loss_multiscale(x[0], x[1], (p01, p10))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            if (step + 1) % max(steps // 5, 1) == 0:
                curve.append(dict(step=step + 1, loss=round(loss.item(), 4), sec=round(time.time() - t0), **epe(net, pairs)))
                print(v, curve[-1], flush=True)
        report[v] = curve
    path = REPO / f"research/results/lacunanet/flow-learning-{'-'.join(variants)}-{steps}.json"
    path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
