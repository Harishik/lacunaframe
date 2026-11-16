"""Verify published assets, decoded media, alignment and summary/report agreement."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main() -> None:
    manifest=json.loads((ROOT/'assets/manifest.json').read_text())
    entries=[manifest['model'],*manifest['videos']]
    candidate=json.loads((ROOT/'assets/candidate.json').read_text())
    if (ROOT/'assets'/candidate['file']).exists():entries.append(candidate)
    for entry in entries:
        assert hashlib.sha256((ROOT/'assets'/entry['file']).read_bytes()).hexdigest()==entry['sha256'],entry['file']
    total_media=0;total_reports=0
    for study in ['natural-v02','checkpoint-v03']:
        folder=ROOT/'examples'/study
        summary=json.loads((folder/'summary.json').read_text())
        for condition in sorted(folder.iterdir()):
            if not condition.is_dir():continue
            report=json.loads((condition/'report.json').read_text())
            count=report['source']['count'] if isinstance(report['source'],dict) else len(report['missing'])
            total_reports+=1
            for method,metric in report['methods'].items():
                assert len(metric['frames'])==count
                assert [row['frame'] for row in metric['frames']]==list(range(count))
                config=report.get('config',report['source'].get('config') if isinstance(report['source'],dict) else None)
                source=report['source']['config']['source'] if isinstance(report['source'],dict) else report['source']
                row=next(r for r in summary['rows'] if r['sequence']==source and r['seed']==config['seed'] and r['method']==method and (r['regime']=='long-gap')==(config['gap']>0))
                assert abs(row['affected_psnr']-metric['affected_psnr'])<1e-10
                assert abs(row['affected_ssim']-metric['affected_ssim'])<1e-10
            for media in condition.glob('*.mp4'):
                cap=cv2.VideoCapture(str(media));decoded=0
                while True:
                    ok,frame=cap.read()
                    if not ok:break
                    assert frame.shape==(180,320,3),str(media)
                    decoded+=1
                cap.release();assert decoded==count,(str(media),decoded,count)
                assert media.stat().st_size>100
                total_media+=1
        for mean in summary['summary']:
            rows=[r for r in summary['rows'] if r['method']==mean['method'] and r['regime']==mean['regime']]
            assert len(rows)==mean['n']
            assert abs(np.mean([r['affected_psnr'] for r in rows])-mean['mean_affected_psnr'])<1e-10
    assert (ROOT/'public/reports/natural-study.json').read_bytes()==(ROOT/'examples/natural-v02/summary.json').read_bytes()
    assert (ROOT/'public/reports/checkpoint-study.json').read_bytes()==(ROOT/'examples/checkpoint-v03/summary.json').read_bytes()
    result=dict(status='passed',verified_asset_hashes=len(entries),decoded_mp4_files=total_media,aligned_condition_reports=total_reports,summary_agreement=True)
    (ROOT/'docs/verification/artifacts.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
