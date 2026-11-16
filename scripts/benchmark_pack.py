"""Lossless, timestamp-checked input/output contract for external VFI comparisons.

Export never sends data anywhere. inputs.npz contains observations only; the
separate targets.npz belongs to the evaluator, not the restoration process.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fastapi.testclient import TestClient
from backend.app import app,ROOT,NATURAL_IDS
from backend.core import corrupt,evaluate


def score(targets: Path, predictions: Path, method: str) -> dict:
    with np.load(targets,allow_pickle=False) as data:
        reference=data['frames'];affected=data['affected'];times=data['timestamps'];fps=float(data['fps'])
    with np.load(predictions,allow_pickle=False) as data:
        result=data['frames'];predicted_times=data['timestamps']
    if predicted_times.shape != times.shape or not np.array_equal(predicted_times,times):
        raise ValueError('Timestamp mismatch: predictions must preserve the exact input timeline')
    if result.dtype != np.uint8 or reference.dtype != np.uint8:
        raise ValueError('RGB uint8 frame arrays required; no implicit rescaling or color conversion')
    if result.shape != reference.shape:
        raise ValueError('Frame dimensions/count mismatch; no automatic resizing or alignment')
    return dict(method=method,targets_sha256=hashlib.sha256(targets.read_bytes()).hexdigest(),
                predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest(),
                timing='External inference runtime not measured by this evaluator',
                metrics=evaluate(reference,result,affected,fps))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    export=commands.add_parser('export');export.add_argument('--source',choices=NATURAL_IDS,default='natural-bmx-trees')
    export.add_argument('--seed',type=int,default=42);export.add_argument('--regime',choices=['scattered','long-gap'],default='long-gap')
    export.add_argument('--out',type=Path,required=True)
    evaluate_command=commands.add_parser('evaluate');evaluate_command.add_argument('--targets',type=Path,required=True)
    evaluate_command.add_argument('--predictions',type=Path,required=True);evaluate_command.add_argument('--method',required=True)
    evaluate_command.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='evaluate':
        report=score(args.targets,args.predictions,args.method)
        with args.out.open('x') as file:json.dump(report,file,indent=2,allow_nan=False)
        print(args.out);return
    # An existing destination is rejected to protect other work and target separation.
    args.out.mkdir(parents=True,exist_ok=False)
    with TestClient(app):
        ref=np.load(ROOT/args.source/'reference.npy')
        metadata=json.loads((ROOT/args.source/'meta.json').read_text());fps=metadata['fps']
    config=dict(seed=args.seed,drop=.2 if args.regime=='scattered' else .1,gap=0 if args.regime=='scattered' else 6,start=.4,occlusion=0,noise=0,blur=0)
    damaged,missing,mask=corrupt(ref,config)
    timestamps=np.arange(len(ref),dtype=np.float64)/fps
    np.savez_compressed(args.out/'inputs.npz',frames=damaged,missing=missing,mask=mask,timestamps=timestamps,fps=fps)
    np.savez_compressed(args.out/'targets.npz',frames=ref,affected=np.broadcast_to(missing[:,None,None],mask.shape),timestamps=timestamps,fps=fps)
    record=dict(protocol='LACUNAFRAME lossless RGB uint8 NHWC v1',source=metadata,config=config,
                target_access='targets.npz is evaluator-only. Never pass it to the external model.',
                output_contract='predictions.npz: frames uint8 NHWC at identical shape and timestamps float64 copied verbatim from inputs.npz')
    (args.out/'protocol.json').write_text(json.dumps(record,indent=2))
    print(args.out)

if __name__=='__main__':main()
