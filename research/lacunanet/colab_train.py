"""LacunaNet training on a Colab GPU (protocol v4, amendment 2).

Needs model.py, model_v2.py, classical_flow.py and data.py next to it in /content.
Training samples are motion-matched to the dev motion histogram (DESIGN_AUDIT section 7).

Usage in Colab (after those files and this one are written to /content):
    !pip -q install lpips remotezip
    !nohup python /content/colab_train.py > /content/train.log 2>&1 &
Checkpoints: /content/ckpt/ckpt-XXXXX.pt every 2,500 iterations, plus /content/ckpt/state.pt (resume).
Same data, sampling, loss and schedule as research/lacunanet/train.py; budget = 60,000 iterations, batch 16.
"""
import io, json, math, os, random, sys, threading, time, urllib.request
from pathlib import Path
import cv2
import numpy as np
import torch
sys.path.insert(0, '/content')
from model import LacunaNet, make_cond, parameter_count
VERSION = int(os.environ.get('LACUNANET_VERSION', '3'))   # 3 = v2 architecture + Lucas-Kanade motion prior
if VERSION >= 2:
    from model_v2 import LacunaNetV2, anchor_loss_final, anchor_loss_multiscale
from data import GoproSource, StratifiedMotion, VimeoStream

# Dev missing-frame bracket motion histogram (research/results/lacunanet/motion-stats.json; statistics only).
MOTION_BINS = [0, 2, 4, 8, 16, 32, 64, 1e9]
MOTION_TARGET = [0.0875, 0.1125, 0.175, 0.2875, 0.175, 0.1375, 0.025]
GOPRO_MIN_SCALE = 0.72

WORK = Path('/content')
CKPT = Path(os.environ.get('LACUNA_CKPT_DIR', WORK / 'ckpt'))   # e.g. a mounted Drive folder, so state survives the runtime
ITERS, BATCH, LR, WD, LPIPS_W, SAVE_EVERY, VIMEO_SWITCH = 60000, 16, 3e-4, 1e-4, 0.1, 2500, 1500
SEED = 20261121   # crop size, outer-anchor probability and sampling live in data.py
ANCHOR_W = 0.5
GOPRO_ZIP = 'https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large.zip'


def to_16x9(img):
    h, w = img.shape[:2]
    if w * 9 > h * 16:
        nw = h * 16 // 9; img = img[:, (w - nw) // 2:(w - nw) // 2 + nw]
    else:
        nh = w * 9 // 16; img = img[(h - nh) // 2:(h - nh) // 2 + nh]
    return cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)


def decode(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def load_gopro():
    """GOPRO_Large train/*/sharp (22 scenes, 2,103 frames) at 640x360, cached in /content/gopro.npz."""
    cache = WORK / 'gopro.npz'
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return list(z['scenes'])
    from remotezip import RemoteZip
    from concurrent.futures import ThreadPoolExecutor
    with RemoteZip(GOPRO_ZIP) as z:
        members = sorted(i.filename for i in z.infolist() if i.filename.startswith('train/') and '/sharp/' in i.filename and i.filename.endswith('.png'))
    names = sorted({m.split('/')[1] for m in members})

    def scene(name):
        with RemoteZip(GOPRO_ZIP) as zz:
            return np.stack([to_16x9(decode(zz.read(m))) for m in members if m.split('/')[1] == name])
    with ThreadPoolExecutor(8) as pool:
        scenes = list(pool.map(scene, names))
    np.savez(cache, scenes=np.array(scenes, dtype=object))
    print('gopro scenes', len(scenes), 'frames', sum(len(s) for s in scenes), flush=True)
    return scenes


def make_sampler(scenes, vimeo, rng):
    """Shared samplers from data.py (no duplicated sampling logic in the trainer)."""
    return StratifiedMotion(GoproSource.from_scenes(scenes), vimeo, MOTION_TARGET, MOTION_BINS, GOPRO_MIN_SCALE, cache=WORK / 'gopro-motion.npy')


def batch(sampler, rng, device):
    frames, conds = [], []
    for _ in range(BATCH):
        c, cond, (m2, p2) = sampler.sample(rng)
        if not m2: c[2] = np.zeros_like(c[2])
        if not p2: c[3] = np.zeros_like(c[3])
        frames.append(np.stack([np.ascontiguousarray(x) for x in c])); conds.append(cond)
    x = torch.from_numpy(np.stack(frames)).to(device).permute(1, 0, 4, 2, 3).float() / 255.
    return x[0], x[1], x[2], x[3], x[4], torch.tensor(conds, dtype=torch.float32, device=device)


def main():
    import lpips
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED); rng = random.Random(SEED)
    CKPT.mkdir(exist_ok=True)
    model = (LacunaNetV2(24, prior='lk' if VERSION == 3 else None) if VERSION >= 2 else LacunaNet(32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.cuda.amp.GradScaler(enabled=device == 'cuda')
    perceptual = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    for p in perceptual.parameters(): p.requires_grad_(False)
    step, gpu_seconds = 0, 0.0
    if (CKPT / 'state.pt').exists():
        st = torch.load(CKPT / 'state.pt', map_location=device)
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); scaler.load_state_dict(st['scaler'])
        step, gpu_seconds = st['step'], st['gpu_seconds']; rng.setstate(st['rng'])
        print('resumed at', step, flush=True)
    scenes = load_gopro()
    vimeo = VimeoStream(WORK / 'vimeo', SEED, start=step // VIMEO_SWITCH)
    sampler = make_sampler(scenes, vimeo, random.Random(SEED + 1))
    print('motion-stratified sampler', json.dumps(sampler.summary()), flush=True)
    print(json.dumps(dict(device=device, gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
                          params=parameter_count(model), gopro_scenes=len(scenes))), flush=True)
    window, t0 = [], time.time()
    while step < ITERS:
        lr = LR * 0.5 * (1 + math.cos(math.pi * step / ITERS))
        for g in opt.param_groups: g['lr'] = lr
        a_m1, a_p1, a_m2, a_p2, target, cond = batch(sampler, rng, device)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device == 'cuda'):
            if VERSION >= 2:
                frame, scale, aux = model(a_m1, a_p1, a_m2, a_p2, cond, return_aux=True)
            else:
                frame, scale = model(a_m1, a_p1, a_m2, a_p2, cond)
        frame, scale = frame.float(), scale.float()
        rec = torch.sqrt((frame - target) ** 2 + 1e-6).mean()
        per = perceptual(frame * 2 - 1, target * 2 - 1).mean()
        resid = (frame.detach() - target).abs().mean(1, keepdim=True)
        nll = (resid / scale + torch.log(scale)).mean()
        loss = rec + LPIPS_W * per + 0.01 * nll
        if VERSION >= 2:
            if VERSION == 3:
                loss = loss + ANCHOR_W * anchor_loss_final(a_m1, a_p1, aux['pyramid'][0][-1].float(), aux['pyramid'][1][-1].float())
            else:
                loss = loss + ANCHOR_W * anchor_loss_multiscale(a_m1, a_p1, tuple([f.float() for f in lv] for lv in aux['pyramid']))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        step += 1
        now = time.time(); gpu_seconds += now - t0; t0 = now
        window.append((rec.item(), per.item(), nll.item()))
        if step % 100 == 0 and window:
            r = np.mean(window, axis=0); window = []
            print(json.dumps(dict(step=step, lr=round(lr, 7), rec=round(r[0], 5), lpips=round(r[1], 5), nll=round(r[2], 4),
                                  hours=round(gpu_seconds / 3600, 3))), flush=True)
        if step % VIMEO_SWITCH == 0:
            vimeo.load_next()
        if step % SAVE_EVERY == 0 or step == ITERS:
            sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(dict(model=sd, c=24 if VERSION >= 2 else 32, version=VERSION, step=step, gpu_hours=gpu_seconds / 3600), CKPT / f'ckpt-{step:05d}.pt')
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step,
                            gpu_seconds=gpu_seconds, rng=rng.getstate()), CKPT / 'state.pt')
            print('saved', step, flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
