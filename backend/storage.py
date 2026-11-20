"""Bounded local experiment storage: UUID folders, metadata, expiry and job state."""
from __future__ import annotations
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from fastapi import HTTPException
from .learned import MANIFEST

ROOT = Path(os.environ.get('LACUNA_DATA', '/tmp/lacunaframe'))
ROOT.mkdir(parents=True, exist_ok=True)
TTL = 3600          # seconds before an idle experiment expires
MAX_FOLDERS = 32    # hard bound on concurrent experiment folders
jobs: dict[str, dict] = {}
gate = threading.Lock()     # one CPU reconstruction at a time
stop = threading.Event()
NATURAL_IDS = [v['id'] for v in MANIFEST['videos']]
DEMO_NAMES = ['Orbital study', 'Crossing paths', 'Drifting textures']
DEMO_IDS = [f'demo{i}' for i in range(len(DEMO_NAMES))]


def save_meta(folder: Path, data: dict) -> None:
    (folder / 'meta.json').write_text(json.dumps(data))


def create() -> tuple[str, Path]:
    cleanup()
    if len(list(ROOT.iterdir())) >= MAX_FOLDERS:
        raise HTTPException(429, 'Session storage full. Restart with a clean data directory or wait one hour.')
    ident = uuid.uuid4().hex
    folder = ROOT / ident
    folder.mkdir()
    return ident, folder


def _protected(name: str) -> bool:
    return name.startswith('demo') or name in NATURAL_IDS


def cleanup() -> None:
    for p in ROOT.iterdir():
        if p.is_dir() and not _protected(p.name) and time.time() - p.stat().st_mtime > TTL:
            if jobs.get(p.name, {}).get('status') != 'running':
                shutil.rmtree(p, ignore_errors=True)
                jobs.pop(p.name, None)


def janitor() -> None:
    while not stop.wait(60):
        cleanup()


def _valid_ident(ident: str) -> bool:
    return ident in NATURAL_IDS + DEMO_IDS or (len(ident) == 32 and all(c in '0123456789abcdef' for c in ident))


def locate(ident: str) -> Path:
    if not _valid_ident(ident):
        raise HTTPException(404, 'Unknown experiment')
    p = ROOT / ident
    if not p.is_dir():
        raise HTTPException(404, 'Experiment expired or does not exist')
    return p


def meta(ident: str) -> dict:
    return json.loads((locate(ident) / 'meta.json').read_text())
