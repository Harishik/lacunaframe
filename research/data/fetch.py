"""Capped, subset-only fetcher for protocol v1 data (see research/protocols/RECOVERABILITY_V1.md §2).

Downloads individual files from pinned Hugging Face revisions, converts them to
compact lossless arrays immediately, deletes the originals, and refuses to run
past the storage cap. Data lives outside git in $LACUNA_EXTERNAL
(default: <repo parent>/lacuna-data). Every source is appended to
research/data/ledger.json with revision, bytes fetched, bytes stored and hashes.

    .venv/bin/python research/data/fetch.py snu-film
    .venv/bin/python research/data/fetch.py vimeo-triplets
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request
import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXTERNAL = Path(os.environ.get('LACUNA_EXTERNAL', REPO.parent / 'lacuna-data')).resolve()
LEDGER = REPO / 'research/data/ledger.json'
CAP_BYTES = int(6 * 1024**3)
SIZE = (640, 360)  # stored width, height (16:9)
HF = 'https://huggingface.co'


def used_bytes() -> int:
    return sum(p.stat().st_size for p in EXTERNAL.rglob('*') if p.is_file()) if EXTERNAL.exists() else 0


def check_cap(incoming: int = 0) -> None:
    if used_bytes() + incoming > CAP_BYTES:
        raise SystemExit(f'Storage cap reached: {used_bytes()/1e9:.2f} GB used + {incoming/1e9:.2f} GB incoming > 6 GiB')


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def get_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def revision(repo: str) -> str:
    return get_json(f'{HF}/api/datasets/{repo}')['sha']


def to_16x9(img: np.ndarray) -> np.ndarray:
    """Centre-crop to 16:9, then area-resize to SIZE. Input/output RGB uint8."""
    h, w = img.shape[:2]
    if w * 9 > h * 16:
        nw = h * 16 // 9
        img = img[:, (w - nw) // 2:(w - nw) // 2 + nw]
    else:
        nh = w * 9 // 16
        img = img[(h - nh) // 2:(h - nh) // 2 + nh]
    return cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)


def decode_png(data: bytes) -> np.ndarray:
    a = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if a is None:
        raise ValueError('PNG decode failed')
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def record(entry: dict) -> None:
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else dict(cap_bytes=CAP_BYTES, external_dir='$LACUNA_EXTERNAL', sources=[])
    ledger['sources'] = [s for s in ledger['sources'] if s['name'] != entry['name']] + [entry]
    ledger['stored_bytes_total'] = sum(s['stored_bytes'] for s in ledger['sources'])
    LEDGER.write_text(json.dumps(ledger, indent=2))


def snu_film(window: int = 48, stride: int = 8) -> None:
    """SNU-FILM test sequences (240 fps): one 48-frame window at stride 8 (= 30 fps) per sequence."""
    repo = 'xypeng9903/SNU_FILM'
    rev = revision(repo)
    out = EXTERNAL / 'snu-film'
    out.mkdir(parents=True, exist_ok=True)
    sequences, fetched = [], 0
    for group in ('GOPRO_test', 'YouTube_test'):
        for seq in get_json(f'{HF}/api/datasets/{repo}/tree/{rev}/test/{group}'):
            name = seq['path'].split('/')[-1]
            target = out / f'{group}-{name}.npz'
            files = sorted(e['path'] for e in get_json(f'{HF}/api/datasets/{repo}/tree/{rev}/{seq["path"]}') if e['path'].endswith('.png'))
            span = (window - 1) * stride + 1
            if len(files) < span:
                stride_used = max(1, (len(files) - 1) // (window - 1))
            else:
                stride_used = stride
            start = (len(files) - ((window - 1) * stride_used + 1)) // 2
            chosen = files[start::stride_used][:window]
            if not target.exists():
                check_cap(40 * 1024**2)
                with ThreadPoolExecutor(8) as pool:
                    blobs = list(pool.map(lambda p: get_bytes(f'{HF}/datasets/{repo}/resolve/{rev}/{p}'), chosen))
                fetched += sum(len(b) for b in blobs)
                frames = np.stack([to_16x9(decode_png(b)) for b in blobs])
                del blobs
                np.savez_compressed(target, frames=frames, source_files=np.array(chosen), stride=stride_used, source_fps=240, fps=240 / stride_used)
            data = np.load(target)
            sequences.append(dict(id=f'{group}-{name}', group=group, file=target.name, frames=int(data['frames'].shape[0]),
                                  stride=int(data['stride']), fps=float(data['fps']), first=str(data['source_files'][0]).split('/')[-1],
                                  sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
            print(sequences[-1]['id'], sequences[-1]['frames'], 'frames', flush=True)
    stored = sum((out / s['file']).stat().st_size for s in sequences)
    record(dict(name='snu-film', repo=f'{HF}/datasets/{repo}', revision=rev, license='SNU-FILM (Choi et al., AAAI 2020); GOPRO subset from Nah et al. CVPR 2017 (CC BY 4.0); YouTube subset research use. Arrays kept local, not redistributed.',
                selection=f'{window}-frame centred window at stride {stride} from each of 31 test sequences; centre-crop 16:9, INTER_AREA to {SIZE[0]}x{SIZE[1]}',
                bytes_fetched_this_run=fetched, stored_bytes=stored, sequences=sequences))
    print(f'snu-film stored {stored/1e6:.1f} MB; external total {used_bytes()/1e9:.2f} GB')


def vimeo_triplets(keep: int = 256, shard: int = 0) -> None:
    """Vimeo90K triplet test: one parquet shard, keep `keep` evenly spaced triplets (native 448x256)."""
    import pyarrow.parquet as pq
    repo = 'paschalc24/vimeo-90k-test-full'
    rev = revision(repo)
    out = EXTERNAL / 'vimeo-triplets'
    out.mkdir(parents=True, exist_ok=True)
    target = out / f'shard{shard}-{keep}.npz'
    path = f'data/test-0000{shard}-of-00004.parquet'
    fetched = 0
    if not target.exists():
        check_cap(500 * 1024**2)
        tmp = out / 'shard.parquet.part'
        urllib.request.urlretrieve(f'{HF}/datasets/{repo}/resolve/{rev}/{path}', tmp)
        fetched = tmp.stat().st_size
        table = pq.read_table(tmp)
        print('columns', table.schema, flush=True)
        cols = table.column_names
        image_cols = [c for c in cols if table.schema.field(c).type.__class__.__name__ in ('StructType',) or 'im' in c.lower() or 'frame' in c.lower()]
        rows = np.linspace(0, table.num_rows - 1, keep).round().astype(int)
        triplets, ids = [], []
        for r in rows:
            row = {c: table.column(c)[int(r)].as_py() for c in cols}
            imgs = []
            for c in image_cols:
                v = row[c]
                blob = v['bytes'] if isinstance(v, dict) else v
                if isinstance(blob, (bytes, bytearray)):
                    imgs.append(decode_png(bytes(blob)))
            if len(imgs) != 3:
                raise SystemExit(f'Unexpected parquet layout: columns {cols}')
            triplets.append(np.stack(imgs))
            ids.append(str(row.get('sequence_id', row.get('id', r))))
        np.savez_compressed(target, triplets=np.stack(triplets), ids=np.array(ids), rows=rows)
        tmp.unlink()
    data = np.load(target)
    record(dict(name='vimeo-triplets', repo=f'{HF}/datasets/{repo}', revision=rev, license='Vimeo90K (Xue et al., IJCV 2019), research use. Arrays kept local.',
                selection=f'{keep} evenly spaced triplets from {path}; native resolution; shard deleted after extraction',
                bytes_fetched_this_run=fetched, stored_bytes=target.stat().st_size, shape=list(data['triplets'].shape),
                sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    print(f'vimeo stored {target.stat().st_size/1e6:.1f} MB; external total {used_bytes()/1e9:.2f} GB')


def vimeo_balanced(per_shard: int = 64) -> None:
    """Vimeo90K triplet test, evenly sampled across all four shards (fixes shard-0 subset bias).

    Each 448 MB shard is downloaded, sampled and deleted before the next one, so peak
    transient storage is one shard.
    """
    import pyarrow.parquet as pq
    repo = 'paschalc24/vimeo-90k-test-full'
    rev = revision(repo)
    out = EXTERNAL / 'vimeo-triplets'
    out.mkdir(parents=True, exist_ok=True)
    target = out / f'balanced-{4 * per_shard}.npz'
    fetched, triplets, ids = 0, [], []
    if not target.exists():
        for shard in range(4):
            check_cap(500 * 1024**2)
            tmp = out / 'shard.parquet.part'
            urllib.request.urlretrieve(f'{HF}/datasets/{repo}/resolve/{rev}/data/test-0000{shard}-of-00004.parquet', tmp)
            fetched += tmp.stat().st_size
            table = pq.read_table(tmp, columns=['im1', 'im2', 'im3', 'sequence_id'])
            for r in np.linspace(0, table.num_rows - 1, per_shard).round().astype(int):
                triplets.append(np.stack([decode_png(table.column(c)[int(r)].as_py()['bytes']) for c in ('im1', 'im2', 'im3')]))
                ids.append(f"{shard}:{table.column('sequence_id')[int(r)].as_py()}")
            del table
            tmp.unlink()
            print('shard', shard, 'sampled', flush=True)
        np.savez_compressed(target, triplets=np.stack(triplets), ids=np.array(ids))
    data = np.load(target)
    record(dict(name='vimeo-balanced', repo=f'{HF}/datasets/{repo}', revision=rev, license='Vimeo90K (Xue et al., IJCV 2019), research use. Arrays kept local.',
                selection=f'{per_shard} evenly spaced triplets from each of the 4 test shards ({4 * per_shard} total); shards deleted after sampling',
                bytes_fetched_this_run=fetched, stored_bytes=target.stat().st_size, shape=list(data['triplets'].shape),
                sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    print(f'vimeo-balanced stored {target.stat().st_size/1e6:.1f} MB; external total {used_bytes()/1e9:.2f} GB')


XIPH = 'https://media.xiph.org/video/derf/y4m/'


def _range(url: str, start: int, length: int) -> bytes:
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{start + length - 1}'})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if len(data) != length:
        raise IOError(f'Short range read from {url}: {len(data)} != {length}')
    return data


def xiph(window: int = 48) -> None:
    """Protocol v2 confirmatory set: centred 48-frame windows from 12 Xiph 1080p sequences via HTTP range reads.

    Only the needed frames are transferred; raw YUV is converted immediately and never stored.
    """
    spec = json.loads((REPO / 'research/protocols/recoverability-v2.json').read_text())['confirmatory_test_set']
    clips = [(c, 2) for c in spec['clips_50fps_stride2']] + [(c, 1) for c in spec['clips_25fps_stride1']]
    out = EXTERNAL / 'xiph'
    out.mkdir(parents=True, exist_ok=True)
    entries, fetched = [], 0
    for name, stride in clips:
        target = out / f'{name}.npz'
        url = XIPH + f'{name}.y4m'
        head = _range(url, 0, 512)
        header = head[:head.index(b'\n') + 1]
        fields = dict((f[:1].decode(), f[1:].decode()) for f in header.split()[1:])
        w, h = int(fields['W']), int(fields['H'])
        if not fields.get('C', '420').startswith('420'):
            raise SystemExit(f'Unsupported chroma in {name}: {fields.get("C")}')
        frame_bytes = w * h * 3 // 2
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
            total = int(r.headers['Content-Length'])
        if head[len(header):len(header) + 6] != b'FRAME\n':
            raise SystemExit(f'Unexpected per-frame header in {name}')
        record_bytes = 6 + frame_bytes
        n = (total - len(header)) // record_bytes
        span = (window - 1) * stride + 1
        start = (n - span) // 2
        indices = [start + k * stride for k in range(window)]
        if not target.exists():
            check_cap(window * record_bytes)
            frames = []
            for i in indices:
                blob = _range(url, len(header) + i * record_bytes, record_bytes)
                if blob[:6] != b'FRAME\n':
                    raise SystemExit(f'Frame marker missing at {name}[{i}]')
                yuv = np.frombuffer(blob[6:], np.uint8).reshape(h * 3 // 2, w)
                frames.append(to_16x9(cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)))
                fetched += record_bytes
            np.savez_compressed(target, frames=np.stack(frames), source_indices=np.array(indices), stride=stride,
                                source_fps=float(fields['F'].split(':')[0]) / float(fields['F'].split(':')[1]))
        entries.append(dict(id=f'xiph-{name}', file=target.name, frames=window, stride=stride, source_indices=[indices[0], indices[-1]],
                            sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        print(name, 'frames', indices[0], '..', indices[-1], flush=True)
    stored = sum((out / e['file']).stat().st_size for e in entries)
    record(dict(name='xiph', repo=XIPH, revision='static files (sizes verified at fetch)', license='Xiph.org derf collection; per-sequence terms (SVT sequences free for research).',
                selection='centred 48-frame windows via HTTP range reads, stride 2 (50 fps) or 1 (25 fps); I420->RGB; 16:9 crop, 640x360',
                bytes_fetched_this_run=fetched, stored_bytes=stored, sequences=entries))
    print(f'xiph stored {stored/1e6:.1f} MB; external total {used_bytes()/1e9:.2f} GB')


def _xiph_window(name: str, window: int, stride_rule) -> tuple[np.ndarray, list[int], str, int]:
    """Range-read a centred `window`-frame window (stride from fps) of one Xiph y4m; returns RGB frames at 640x360."""
    url = XIPH + f'{name}.y4m'
    head = _range(url, 0, 512)
    header = head[:head.index(b'\n') + 1]
    fields = dict((f[:1].decode(), f[1:].decode()) for f in header.split()[1:])
    w, h, chroma = int(fields['W']), int(fields['H']), fields.get('C', '420')
    num, den = (float(v) for v in fields['F'].split(':'))
    stride = stride_rule(num / den)
    if chroma.startswith('420'):
        frame_bytes = w * h * 3 // 2
    elif chroma.startswith('422'):
        frame_bytes = w * h * 2
    else:
        raise SystemExit(f'Unsupported chroma {chroma} in {name}')
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
        total = int(r.headers['Content-Length'])
    record_bytes = 6 + frame_bytes
    n = (total - len(header)) // record_bytes
    span = (window - 1) * stride + 1
    start = (n - span) // 2
    indices = [start + k * stride for k in range(window)]
    check_cap(window * record_bytes)
    frames = []
    for i in indices:
        blob = _range(url, len(header) + i * record_bytes, record_bytes)
        if blob[:6] != b'FRAME\n':
            raise SystemExit(f'Frame marker missing at {name}[{i}]')
        raw = np.frombuffer(blob[6:], np.uint8)
        y = raw[:w * h]
        if chroma.startswith('422'):
            u = raw[w * h:w * h + (w // 2) * h].reshape(h, w // 2).astype(np.uint16)
            v = raw[w * h + (w // 2) * h:].reshape(h, w // 2).astype(np.uint16)
            u = ((u[0::2] + u[1::2] + 1) // 2).astype(np.uint8)
            v = ((v[0::2] + v[1::2] + 1) // 2).astype(np.uint8)
            yuv = np.concatenate([y, u.reshape(-1), v.reshape(-1)]).reshape(h * 3 // 2, w)
        else:
            yuv = raw.reshape(h * 3 // 2, w)
        frames.append(to_16x9(cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)))
    return np.stack(frames), indices, chroma, stride


def xiph_c(window: int = 48) -> None:
    """Protocol v4 untouched set: centred 48-frame windows; stride 2 for >= 50 fps sources (<= 30 fps effective)."""
    spec = json.loads((REPO / 'research/protocols/recoverability-v4.json').read_text())['eval_sets']['xiph_c']
    out = EXTERNAL / 'xiph-c'
    out.mkdir(parents=True, exist_ok=True)
    entries, fetched = [], 0
    for name in spec:
        target = out / f'{name}.npz'
        if not target.exists():
            frames, indices, chroma, stride = _xiph_window(name, window, lambda fps: 2 if fps >= 49 else 1)
            np.savez_compressed(target, frames=frames, source_indices=np.array(indices), stride=stride, chroma=chroma)
            fetched += frames.nbytes
        z = np.load(target)
        entries.append(dict(id=f'xiphc-{name}', file=target.name, frames=window, stride=int(z['stride']), chroma=str(z['chroma']),
                            source_indices=[int(z['source_indices'][0]), int(z['source_indices'][-1])], sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        print(name, entries[-1]['chroma'], 'stride', entries[-1]['stride'], flush=True)
    stored = sum((out / e['file']).stat().st_size for e in entries)
    record(dict(name='xiph-c', repo=XIPH, revision='static files', license='Xiph.org derf collection; per-sequence terms (incl. Blender Foundation CC BY for Elephants Dream / Sintel).',
                selection='protocol v4: centred 48-frame windows, stride 2 for >=50 fps, I420 conversion, 16:9 crop, 640x360',
                bytes_fetched_this_run=fetched, stored_bytes=stored, sequences=entries))
    print(f'xiph-c stored {stored/1e6:.1f} MB; external total {used_bytes()/1e9:.2f} GB')


def xiph_b(window: int = 48) -> None:
    """Protocol v3 untouched set: centred 48-frame windows (stride 1, 30 fps) from 10 Xiph 1080p sequences.

    Handles 4:2:0 and planar 4:2:2 (chroma averaged vertically to 4:2:0, then the same
    OpenCV I420 conversion as Xiph v2). Raw YUV is never stored.
    """
    spec = json.loads((REPO / 'research/protocols/recoverability-v3.json').read_text())['test_set_xiph_b']
    out = EXTERNAL / 'xiph-b'
    out.mkdir(parents=True, exist_ok=True)
    entries, fetched = [], 0
    for name in spec['clips']:
        target = out / f'{name}.npz'
        url = XIPH + f'{name}.y4m'
        head = _range(url, 0, 512)
        header = head[:head.index(b'\n') + 1]
        fields = dict((f[:1].decode(), f[1:].decode()) for f in header.split()[1:])
        w, h, chroma = int(fields['W']), int(fields['H']), fields.get('C', '420')
        if chroma.startswith('420'):
            frame_bytes = w * h * 3 // 2
        elif chroma.startswith('422'):
            frame_bytes = w * h * 2
        else:
            raise SystemExit(f'Unsupported chroma {chroma} in {name}')
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
            total = int(r.headers['Content-Length'])
        record_bytes = 6 + frame_bytes
        n = (total - len(header)) // record_bytes
        start = (n - window) // 2
        indices = list(range(start, start + window))
        if not target.exists():
            check_cap(window * record_bytes)
            frames = []
            for i in indices:
                blob = _range(url, len(header) + i * record_bytes, record_bytes)
                if blob[:6] != b'FRAME\n':
                    raise SystemExit(f'Frame marker missing at {name}[{i}]')
                raw = np.frombuffer(blob[6:], np.uint8)
                y = raw[:w * h].reshape(h, w)
                if chroma.startswith('422'):
                    u = raw[w * h:w * h + (w // 2) * h].reshape(h, w // 2).astype(np.uint16)
                    v = raw[w * h + (w // 2) * h:].reshape(h, w // 2).astype(np.uint16)
                    u = ((u[0::2] + u[1::2] + 1) // 2).astype(np.uint8)
                    v = ((v[0::2] + v[1::2] + 1) // 2).astype(np.uint8)
                    yuv = np.concatenate([y.reshape(-1), u.reshape(-1), v.reshape(-1)]).reshape(h * 3 // 2, w)
                else:
                    yuv = raw.reshape(h * 3 // 2, w)
                frames.append(to_16x9(cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)))
                fetched += record_bytes
            np.savez_compressed(target, frames=np.stack(frames), source_indices=np.array(indices), stride=1, chroma=chroma)
        entries.append(dict(id=f'xiphb-{name}', file=target.name, frames=window, chroma=chroma, source_indices=[indices[0], indices[-1]],
                            sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        print(name, chroma, 'frames', indices[0], '..', indices[-1], flush=True)
    stored = sum((out / e['file']).stat().st_size for e in entries)
    record(dict(name='xiph-b', repo=XIPH, revision='static files', license='Xiph.org derf collection; per-sequence terms.',
                selection='protocol v3: centred 48-frame windows, stride 1 (30 fps), 4:2:2 averaged to 4:2:0, I420->RGB, 16:9 crop, 640x360',
                bytes_fetched_this_run=fetched, stored_bytes=stored, sequences=entries))
    print(f'xiph-b stored {stored/1e6:.1f} MB; external total {used_bytes()/1e9:.2f} GB')


def gopro_train() -> None:
    """LacunaNet training data (protocol v4): GOPRO_Large *train* sharp frames, range-extracted from the zip.

    Only train/<scene>/sharp/*.png members are transferred; each scene is downscaled to 640x360
    (16:9 already) and stored as one uint8 array. Test scenes are never touched.
    """
    from remotezip import RemoteZip
    url = 'https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large.zip'
    out = EXTERNAL / 'train' / 'gopro'
    out.mkdir(parents=True, exist_ok=True)
    fetched, entries = 0, []
    with RemoteZip(url) as z:
        members = sorted(i.filename for i in z.infolist() if i.filename.startswith('train/') and '/sharp/' in i.filename and i.filename.endswith('.png'))
    scenes = sorted({m.split('/')[1] for m in members})

    def one(scene: str) -> tuple[str, int, int]:
        target = out / f'{scene}.npz'
        names = [m for m in members if m.split('/')[1] == scene]
        got = 0
        if not target.exists():
            check_cap(len(names) * 2 * 1024**2)
            frames = []
            with RemoteZip(url) as zz:                      # one connection per worker (RemoteZip is not thread-safe)
                for m in names:
                    data = zz.read(m)
                    got += len(data)
                    frames.append(to_16x9(decode_png(data)))
            tmp = target.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, frames=np.stack(frames), names=np.array(names))
            tmp.rename(target)
        print(scene, len(names), flush=True)
        return scene, len(names), got

    with ThreadPoolExecutor(6) as pool:
        done = list(pool.map(one, scenes))
    for scene, count, got in done:
        fetched += got
        target = out / f'{scene}.npz'
        entries.append(dict(scene=scene, frames=count, file=target.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    stored = sum((out / e['file']).stat().st_size for e in entries)
    record(dict(name='gopro-train', repo=url, revision='mirror of GOPRO_Large (Nah et al., CVPR 2017, CC BY 4.0)', license='CC BY 4.0',
                selection='train/*/sharp only (22 scenes, 2,103 frames), 640x360; GOPRO test scenes never transferred',
                bytes_fetched_this_run=fetched, stored_bytes=stored, sequences=entries))
    print(f'gopro-train stored {stored/1e6:.1f} MB; external total {used_bytes()/1e9:.2f} GB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=['snu-film', 'vimeo-triplets', 'vimeo-balanced', 'xiph', 'xiph-b', 'xiph-c', 'gopro-train'])
    args = parser.parse_args()
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    {'snu-film': snu_film, 'vimeo-triplets': vimeo_triplets, 'vimeo-balanced': vimeo_balanced, 'xiph': xiph, 'xiph-b': xiph_b, 'gopro-train': gopro_train, 'xiph-c': xiph_c}[args.source]()
