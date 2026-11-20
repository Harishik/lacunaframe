"""Background reconstruction job: restore, evaluate, encode and write the report."""
from __future__ import annotations
import csv
import importlib.metadata
import json
import logging
import platform
import threading
import time
from pathlib import Path
import cv2
import numpy as np
import psutil
from . import storage
from .core import evaluate, restore
from .learned import model_info, session
from .media import encode

log = logging.getLogger('lacunaframe')
REPORT_VERSION = '0.3.0'
METRIC_NOTES = ('PSNR from pooled MSE; Infinity string means exact match. SSIM is mean local RGB SSIM; masked SSIM includes '
                'neighborhoods outside mask. Temporal error = mean absolute error in consecutive-frame differences / 255. '
                'Missing corrupted frames use zero placeholders for metric penalty only.')


def _environment() -> dict:
    return dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__,
                skimage=importlib.metadata.version('scikit-image'))


def _missing_display(damaged: np.ndarray, missing: np.ndarray) -> np.ndarray:
    display = damaged.copy()
    for i in np.flatnonzero(missing):
        display[i][:] = (30, 28, 35)
        cv2.putText(display[i], 'MISSING SAMPLE', (43, 94), cv2.FONT_HERSHEY_SIMPLEX, .6, (250, 140, 120), 1, cv2.LINE_AA)
    return display


def _run_method(ident: str, method: str, j: int, total: int, damaged, missing, mask):
    """Restore with one method while sampling whole-process RSS every 20 ms."""
    storage.jobs[ident]['stage'] = 'Loading ' + method
    loading = time.perf_counter()
    if method.startswith('rife'):
        session()
    loading = time.perf_counter() - loading
    memory = [psutil.Process().memory_info().rss]
    sampling = threading.Event()

    def sample_memory():
        while not sampling.wait(.02):
            memory.append(psutil.Process().memory_info().rss)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    begin = time.perf_counter()

    def update(fraction):
        storage.jobs[ident]['progress'] = (j + fraction) / total * .9

    storage.jobs[ident]['stage'] = 'Restoring with ' + method
    try:
        out, warnings, uncertainty = restore(damaged, missing, mask, method, update)
        runtime = time.perf_counter() - begin
    finally:
        sampling.set()
        sampler.join()
    return out, warnings, uncertainty, runtime, loading, max(memory)


def _write_csv(path: Path, report: dict) -> None:
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'frame', 'time', 'mse', 'psnr', 'ssim', 'affected_psnr', 'affected_ssim'])
        writer.writeheader()
        for method, result in {'corrupted': report['corrupted'], **report['methods']}.items():
            writer.writerows(dict(method=method, **row) for row in result['frames'])


def process(ident: str, source: Path, methods: list[str]) -> None:
    p = storage.locate(ident)
    try:
        ref, damaged = np.load(source / 'reference.npy'), np.load(source / 'corrupted.npy')
        missing, mask = np.load(source / 'missing.npy'), np.load(source / 'mask.npy')
        info = json.loads((p / 'meta.json').read_text())
        fps = info['fps']
        affected = mask.copy()
        affected[missing] = True
        if info['config']['noise'] or info['config']['blur']:
            affected[:] = True
        report = dict(version=REPORT_VERSION, environment=_environment(), source=info, metric_notes=METRIC_NOTES,
                      corrupted=evaluate(ref, damaged, affected, fps), methods={})
        report['corrupted']['media'] = encode(p, 'corrupted', _missing_display(damaged, missing), fps)
        for j, method in enumerate(methods):
            out, warnings, uncertainty, runtime, loading, peak = _run_method(ident, method, j, len(methods), damaged, missing, mask)
            storage.jobs[ident]['stage'] = 'Evaluating and encoding ' + method
            np.save(p / f'{method}.npy', out)
            metrics = evaluate(ref, out, affected, fps)
            media = encode(p, method, out, fps)
            report['methods'][method] = dict(
                **metrics, runtime_seconds=runtime, model_initialization_seconds=loading, process_peak_rss_mib=peak / 1024**2,
                model=model_info() if method.startswith('rife') else None, inference_scale=.5 if method == 'rife_fast' else 1.,
                warnings=warnings, reliability_available=method == 'flow', uncertainty=uncertainty if method == 'flow' else None, media=media)
        (p / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        _write_csv(p / 'metrics.csv', report)
        storage.jobs[ident].update(status='complete', progress=1)
        log.info(json.dumps(dict(event='run_complete', id=ident, methods=methods)))
    except Exception:
        log.exception(json.dumps(dict(event='run_failed', id=ident)))
        storage.jobs[ident].update(status='failed', error='Processing failed; inspect backend logs.')
    finally:
        storage.gate.release()
