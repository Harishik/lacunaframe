"""Explicit, checksum-verified setup. Inference never downloads or uploads data."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
ROOT=Path(__file__).resolve().parents[1]/'assets'
manifest=json.loads((ROOT/'manifest.json').read_text())
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--candidate',action='store_true',help='Also fetch the optional RIFE 4.26 research checkpoint')
args=parser.parse_args()
entries=[manifest['model'],*manifest['videos']]
if args.candidate: entries.append(json.loads((ROOT/'candidate.json').read_text()))
for entry in entries:
    path=ROOT/entry['file'];path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']:
        print('Verified',entry['file']);continue
    temporary=path.with_suffix('.download')
    try:
        urllib.request.urlretrieve(entry['url'],temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Checksum mismatch: '+entry['file'])
        temporary.replace(path)
        print('Downloaded and verified',entry['file'])
    finally:temporary.unlink(missing_ok=True)
