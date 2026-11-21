"""Recover LacunaNet checkpoints from a Colab notebook downloaded via the Google Drive connector.

Colab cells print blocks of the form
    CKPT <name> <sha256-of-raw-bytes> <B64START>base64(zlib(raw))<B64END>
The notebook (saved to Drive) is fetched with the Drive connector, which stores it as a local
file. This script extracts every block, verifies the SHA-256 and writes the raw checkpoints to
$LACUNA_EXTERNAL/models/lacunanet/.

    .venv/bin/python research/lacunanet/colab_pull.py <downloaded-notebook-file>
"""
import base64
import hashlib
import json
import os
import re
import sys
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('LACUNA_EXTERNAL', REPO.parent / 'lacuna-data')) / 'models' / 'lacunanet'


def outputs_text(nb: dict) -> str:
    parts = []
    for cell in nb['cells']:
        for o in cell.get('outputs', []):
            t = o.get('text', '')
            parts.append(''.join(t) if isinstance(t, list) else t)
    return '\n'.join(parts)


def main() -> None:
    nb = json.loads(Path(sys.argv[1]).read_text())
    text = outputs_text(nb)
    OUT.mkdir(parents=True, exist_ok=True)
    found = 0
    for name, sha, blob in re.findall(r'CKPT (\S+) ([0-9a-f]{64}) B64START(.*?)B64END', text, re.S):
        raw = zlib.decompress(base64.b64decode(blob))
        ok = hashlib.sha256(raw).hexdigest() == sha
        if not ok:
            print('HASH MISMATCH', name)
            continue
        (OUT / name).write_bytes(raw)
        found += 1
        print('saved', name, len(raw), 'bytes, sha256 ok')
    log = re.findall(r'(\{"step": \d+.*?\})', text)
    if log:
        (OUT / 'colab-train-log-tail.jsonl').write_text('\n'.join(log) + '\n')
    print(found, 'checkpoint(s) recovered')


if __name__ == '__main__':
    main()
