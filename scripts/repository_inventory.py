"""Inventory the Git payload, excluding this generated manifest itself."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'docs/verification/repository-inventory.json'


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    paths=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    rows=[]
    for name in sorted(filter(None,paths)):
        p=ROOT/name
        if p==DEST:continue
        rows.append(dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    data=dict(scope='Git-tracked payload at publication; excludes this manifest and Git metadata',files=len(rows),total_bytes=sum(r['bytes'] for r in rows),entries=rows)
    if args.check:
        assert json.loads(DEST.read_text())==data,'Versioned payload differs; regenerate repository_inventory.py and review changes'
        print('Inventory matches',len(rows),'files')
    else:
        DEST.write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps(dict(files=data['files'],bytes=data['total_bytes'])))

if __name__=='__main__':main()
