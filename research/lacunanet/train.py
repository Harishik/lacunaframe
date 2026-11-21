"""CPU training of LacunaNet (protocol v4 §4). Resumable; checkpoints every 2 wall-clock hours.

    ../.venv-train/bin/python research/lacunanet/train.py [--hours 48] [--threads 6]

Writes $LACUNA_EXTERNAL/models/lacunanet/ckpt-XXh.pt, train-log.jsonl and state.pt (for resume).
Elapsed training time is accumulated across restarts, so the 48-hour budget is total CPU training time.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch
import torch.nn.functional as F
from research.lacunanet.data import GoproSource, VimeoStream, batch
from research.lacunanet.model import LacunaNet, parameter_count

EXTERNAL = Path(os.environ.get('LACUNA_EXTERNAL', REPO.parent / 'lacuna-data')).resolve()
OUT = Path(os.environ.get('LACUNA_TRAIN_OUT', EXTERNAL / 'models' / 'lacunanet'))
WIDTH, BATCH, LR, WD, LPIPS_W = 32, 8, 3e-4, 1e-4, 0.1
VIMEO_SWITCH = 1500   # iterations per streamed Vimeo shard
SEED = 20261121


def charbonnier(x, y, eps=1e-3):
    return torch.sqrt((x - y) ** 2 + eps ** 2).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=48.0)
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    import lpips
    perceptual = lpips.LPIPS(net='alex', verbose=False).eval()
    for p in perceptual.parameters():
        p.requires_grad_(False)
    model = LacunaNet(WIDTH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    OUT.mkdir(parents=True, exist_ok=True)
    state_path, log_path = OUT / 'state.pt', OUT / 'train-log.jsonl'
    elapsed, step, next_ckpt = 0.0, 0, 2.0
    if state_path.exists():
        st = torch.load(state_path, map_location='cpu')
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
        elapsed, step, next_ckpt = st['elapsed'], st['step'], st['next_ckpt']
        rng.setstate(st['rng'])
        print(f'resumed at step {step}, {elapsed/3600:.2f} h', flush=True)
    budget = args.hours * 3600
    gopro = GoproSource(EXTERNAL / 'train' / 'gopro')
    vimeo = VimeoStream(EXTERNAL / 'train' / 'vimeo-cache', SEED + step // VIMEO_SWITCH)
    print(f'LacunaNet C={WIDTH}: {parameter_count(model)} parameters; {len(gopro.scenes)} GOPRO scenes', flush=True)
    window, t_last = [], time.perf_counter()
    while elapsed < budget:
        lr = LR * 0.5 * (1 + math.cos(math.pi * min(elapsed / budget, 1.0)))
        for g in opt.param_groups:
            g['lr'] = lr
        a_m1, a_p1, a_m2, a_p2, target, cond = batch(gopro, vimeo, BATCH, rng)
        frame, scale = model(a_m1, a_p1, a_m2, a_p2, cond)
        rec = charbonnier(frame, target)
        per = perceptual(frame * 2 - 1, target * 2 - 1).mean()
        resid = (frame.detach() - target).abs().mean(1, keepdim=True)
        nll = (resid / scale + torch.log(scale)).mean()
        loss = rec + LPIPS_W * per + 0.01 * nll
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        now = time.perf_counter()
        elapsed += now - t_last
        t_last = now
        window.append(dict(rec=rec.item(), lpips=per.item(), nll=nll.item()))
        if step % 100 == 0:
            avg = {k: sum(w[k] for w in window) / len(window) for k in window[0]}
            line = dict(step=step, hours=round(elapsed / 3600, 3), lr=lr, **{k: round(v, 5) for k, v in avg.items()})
            with log_path.open('a') as f:
                f.write(json.dumps(line) + '\n')
            print(line, flush=True)
            window = []
        if step % VIMEO_SWITCH == 0:
            vimeo.load_next()
        if elapsed / 3600 >= next_ckpt or elapsed >= budget:
            tag = f'{min(next_ckpt, args.hours):05.1f}h'
            torch.save(dict(model=model.state_dict(), c=WIDTH, step=step, hours=elapsed / 3600), OUT / f'ckpt-{tag}.pt')
            next_ckpt += 2.0
        if step % 200 == 0 or elapsed >= budget:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), elapsed=elapsed, step=step,
                            next_ckpt=next_ckpt, rng=rng.getstate()), state_path)
    print('training budget reached', flush=True)


if __name__ == '__main__':
    main()
