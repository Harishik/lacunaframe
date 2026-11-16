"""Check repository documentation links, embedded study data and visual provenance."""
from pathlib import Path
import json
import re
from urllib.parse import unquote,urlsplit
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]


def main() -> None:
    files=[ROOT/'README.md',ROOT/'CONTRIBUTING.md',ROOT/'CHANGELOG.md']
    for directory in ['docs','examples','assets']:
        files.extend((ROOT/directory).rglob('*.md'))
    files.append(ROOT/'docs/experience/index.html')
    failures=[];links=0
    for file in files:
        content=file.read_text()
        # Code listings are examples, not clickable file links.
        content=re.sub(r'```.*?```','',content,flags=re.S)
        targets=re.findall(r'\]\(([^)]+)\)',content) if file.suffix=='.md' else re.findall(r'(?:href|src)="([^"]+)"',content)
        for target in targets:
            target=target.strip().strip('<>').split(' "')[0]
            parsed=urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:continue
            path=(ROOT/parsed.path.lstrip('/')) if parsed.path.startswith('/') else file.parent/unquote(parsed.path)
            links+=1
            if not path.exists():failures.append(f'{file.relative_to(ROOT)}: {target}')
    html=(ROOT/'docs/experience/index.html').read_text()
    embedded=re.search(r'<script id="study-data" type="application/json">(.*?)</script>',html,re.S)
    assert embedded,'Missing embedded study payload'
    data=json.loads(embedded[1])
    for name,path in [('natural','examples/natural-v02/summary.json'),('checkpoints','examples/checkpoint-v03/summary.json'),('example','docs/visuals/example.json')]:
        assert data[name]==json.loads((ROOT/path).read_text()),f'Stale embedded {name}'
    example=data['example'];missing=example['missing']
    for method in ['linear','flow','rife']:
        measured=next(r for r in data['natural']['rows'] if r['sequence']=='natural-bmx-trees' and r['regime']=='long-gap' and r['seed']==42 and r['method']==method)
        assert abs(example['metrics'][method]['affected_psnr']-measured['affected_psnr'])<1e-9,f'Documentation differs from preserved {method} experiment'
    worst=max((r for r in example['metrics']['rife']['frames'] if missing[r['frame']]),key=lambda r:r['mse'])
    assert worst['frame']==example['worst_rife_frame'],'Contact sheet frame selection is inconsistent'
    with Image.open(ROOT/'docs/visuals/recovery-loop.gif') as gif:
        assert gif.n_frames==example['count'] and gif.size==(960,236)
    assert not failures,'Broken local links:\n'+'\n'.join(failures)
    result=dict(status='passed',documentation_files=len(files),local_links=links,embedded_data_matches=True,gif_frames=example['count'],worst_frame=example['worst_rife_frame'])
    (ROOT/'docs/verification/documentation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
