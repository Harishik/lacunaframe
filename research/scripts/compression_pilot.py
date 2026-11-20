"""Actual CPU quantization feasibility, with scene-disjoint calibration.

Builds models with ORT's existing quantizer; no algorithmic novelty or training
is implied. Each evaluated variant runs in a fresh subprocess. Unsupported
operators are retained as failures, not replaced by a different method.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from backend import decode
from backend.app import normalize
from backend.core import evaluate
PROTOCOL=json.loads((ROOT/'research/protocols/compression-pilot.json').read_text())
ARCHIVE=ROOT/'research/results/compression-pilot'
# New trials must write elsewhere so the archived evidence is never overwritten.
# The variable is inherited by build/worker subprocesses and the profiler/summary.
OUT=Path(os.environ.get('LACUNA_PILOT_OUT',ARCHIVE)).resolve()
MODELS=OUT/'models'


def peak_rss_mib(maxrss: int|None=None,system: str|None=None) -> float:
    """Convert ru_maxrss to MiB. Linux reports KiB; macOS reports bytes."""
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if maxrss is None else maxrss
    return value/1024**2 if (system or platform.system())=='Darwin' else value/1024


def host_metadata() -> dict:
    """CPU, memory and power provenance for Linux (proc/cgroup) and macOS (sysctl/pmset)."""
    data=dict(platform=platform.platform(),python=platform.python_version(),logical_cpus=os.cpu_count(),
              packages={n:importlib.metadata.version(n) for n in ['numpy','onnx','onnxruntime','opencv-python-headless','scikit-image']})
    def run(*cmd: str) -> str:
        try:return subprocess.run(cmd,capture_output=True,text=True,timeout=10).stdout.strip()
        except (OSError,subprocess.TimeoutExpired):return ''
    if platform.system()=='Darwin':
        data['cpu_model']=run('sysctl','-n','machdep.cpu.brand_string') or 'unknown'
        data['physical_cpus']=run('sysctl','-n','hw.physicalcpu') or 'unknown'
        memory=run('sysctl','-n','hw.memsize');data['memory_bytes']=int(memory) if memory.isdigit() else 'unknown'
        power=run('pmset','-g')
        data['low_power_mode']=next((line.split()[-1] for line in power.splitlines() if 'lowpowermode' in line),'unknown')
        data['thermal']=run('pmset','-g','therm') or 'unknown'
        data['cpu_quota']=data['memory_limit']='not applicable (macOS)'
    else:
        cpu=Path('/proc/cpuinfo')
        data['cpu_model']=next((line.split(':',1)[1].strip() for line in cpu.read_text().splitlines() if line.startswith('model name')),'unknown') if cpu.exists() else 'unknown'
        for name,path in [('cpu_quota','/sys/fs/cgroup/cpu.max'),('memory_limit','/sys/fs/cgroup/memory.max')]:
            p=Path(path);data[name]=p.read_text().strip() if p.exists() else 'unknown'
    return data


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frames(scene: str) -> np.ndarray:
    decoded,_=decode.sampled(ROOT/f'assets/video/{scene}.mp4',every=2,limit=48)
    items=[normalize(a) for a in decoded]
    if not items:raise ValueError(f'No decoded frames: {scene}')
    return np.stack(items)


def pairs(scene: str,starts: list[int]) -> list[dict]:
    source=frames(scene);result=[]
    for gap in PROTOCOL['anchor_separations']:
        for start in starts:
            if start+gap>=len(source):continue
            middle=start+gap//2
            result.append(dict(scene=scene,left=start,right=start+gap,target=middle,t=.5,a=source[start],b=source[start+gap],reference=source[middle]))
    return result


def feeds(a: np.ndarray,b: np.ndarray,t: float,variant: str) -> dict[str,np.ndarray]:
    if variant=='rife426-fp32':
        from backend.candidate import pack
        return {'input':pack(a,b,t)}
    if variant=='rife49-half':
        h,w=a.shape[:2];a=cv2.resize(a,(w//2,h//2),interpolation=cv2.INTER_AREA);b=cv2.resize(b,(w//2,h//2),interpolation=cv2.INTER_AREA)
    h,w=a.shape[:2]
    def tensor(x: np.ndarray) -> np.ndarray:
        return np.pad(x.astype(np.float32).transpose(2,0,1)[None]/255,((0,0),(0,0),(0,(-h)%32),(0,(-w)%32)))
    return dict(img0=tensor(a),img1=tensor(b),timestep=np.array([t],np.float32))


def path_for(variant: str) -> Path:
    if variant=='rife426-fp32':return ROOT/'assets/models/rife426.onnx'
    if variant in ['rife49-fp32','rife49-half']:return ROOT/'assets/models/rife49.onnx'
    return MODELS/f'{variant}.onnx'


def build(variant: str) -> dict:
    from onnxruntime.quantization import CalibrationDataReader,QuantType,QuantFormat,quantize_dynamic,quantize_static
    original=ROOT/'assets/models/rife49.onnx';target=path_for(variant)
    config=dict(variant=variant,input_sha256=sha(original),protocol_sha256=sha(ROOT/'research/protocols/compression-pilot.json'),explicit_quant_pre_process=False)
    if variant=='rife49-dynamic-u8':
        quantize_dynamic(original,target,op_types_to_quantize=['Conv'],weight_type=QuantType.QUInt8,per_channel=False)
    else:
        calibration=pairs(PROTOCOL['calibration_scene'],PROTOCOL['calibration_starts'])
        class Reader(CalibrationDataReader):
            def __init__(self):self.items=iter([feeds(p['a'],p['b'],p['t'],'rife49-fp32') for p in calibration])
            def get_next(self):return next(self.items,None)
        # Calibration uses ORT's internal CPU session/default thread settings.
        # Measured inference workers below explicitly use two threads.
        extra={'ActivationSymmetric':True,'WeightSymmetric':True}
        selected=None
        if variant=='rife49-qdq-trunk':
            selected=[n.name for n in onnx.load(original).graph.node if n.op_type=='Conv' and '/convblock/' in n.name]
        quantize_static(original,target,Reader(),quant_format=QuantFormat.QDQ,op_types_to_quantize=['Conv'],nodes_to_quantize=selected,per_channel=True,activation_type=QuantType.QInt8,weight_type=QuantType.QInt8,calibration_providers=['CPUExecutionProvider'],extra_options=extra)
        config['calibration_pairs']=[{k:p[k] for k in ['scene','left','right','target','t']} for p in calibration]
        config['quantized_node_names']=selected if selected is not None else [n.name for n in onnx.load(original).graph.node if n.op_type=='Conv']
    onnx.checker.check_model(target)
    config.update(output_sha256=sha(target),model_file_bytes=target.stat().st_size,operator_counts=dict(Counter(n.op_type for n in onnx.load(target).graph.node)))
    (MODELS/f'{variant}.json').write_text(json.dumps(config,indent=2))
    return config


def worker(variant: str) -> None:
    path=path_for(variant);options=ort.SessionOptions();options.intra_op_num_threads=PROTOCOL['threads'];options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level=3
    began=time.perf_counter();session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']);load=time.perf_counter()-began
    assert session.get_providers()==['CPUExecutionProvider']
    cases=[p for scene in PROTOCOL['evaluation_scenes'] for p in pairs(scene,PROTOCOL['evaluation_starts'])]
    first=cases[0]
    for _ in range(PROTOCOL['warmup_calls']):session.run(None,feeds(first['a'],first['b'],first['t'],variant))
    rows=[];outputs=[]
    for case in cases:
        h,w=case['a'].shape[:2];durations=[]
        # Timed scope includes packing, inference, crop, conversion and resize.
        for _ in range(PROTOCOL['timed_repeats']):
            start=time.perf_counter();inputs=feeds(case['a'],case['b'],case['t'],variant)
            value=session.run(None,inputs)[0]
            if not np.isfinite(value).all():raise RuntimeError('Nonfinite model output')
            hh,ww=(h//2,w//2) if variant=='rife49-half' else (h,w)
            image=value[0,:,:hh,:ww].transpose(1,2,0)
            if variant=='rife49-half':image=cv2.resize(image,(w,h),interpolation=cv2.INTER_LINEAR)
            image=np.clip(image*255,0,255).round().astype(np.uint8)
            durations.append(time.perf_counter()-start)
        metric=evaluate(case['reference'][None],image[None],np.ones((1,h,w),bool),12)
        rows.append(dict(**{k:case[k] for k in ['scene','left','right','target','t']},variant=variant,psnr=metric['psnr'],ssim=metric['ssim'],median_seconds=float(np.median(durations)),p95_seconds=float(np.percentile(durations,95)),duration_samples=durations))
        outputs.append(image)
    np.save(OUT/f'{variant}-predictions.npy',np.stack(outputs))
    data=dict(status='complete',variant=variant,provider=session.get_providers(),load_seconds=load,model_sha256=sha(path),model_file_bytes=path.stat().st_size,process_peak_rss_mib=peak_rss_mib(),rows=rows)
    (OUT/f'{variant}.json').write_text(json.dumps(data,indent=2,allow_nan=False))


def main() -> None:
    global OUT,MODELS
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker');parser.add_argument('--build')
    parser.add_argument('--out',type=Path,help='fresh trial directory; must not exist or must be empty')
    args=parser.parse_args()
    if args.out:
        target=args.out.resolve()
        if target==ARCHIVE.resolve():raise SystemExit('Refusing to overwrite the archived pilot; choose a new --out directory')
        if target.exists() and any(target.iterdir()):raise SystemExit(f'Refusing to reuse non-empty trial directory: {target}')
        os.environ['LACUNA_PILOT_OUT']=str(target);OUT=target;MODELS=OUT/'models'
    OUT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(exist_ok=True)
    if args.worker:worker(args.worker);return
    if args.build:build(args.build);return
    manifest=json.loads((ROOT/'assets/manifest.json').read_text())
    candidate=json.loads((ROOT/'assets/candidate.json').read_text())
    for entry in [manifest['model'],*manifest['videos'],candidate]:
        assert sha(ROOT/'assets'/entry['file'])==entry['sha256'],f'Input checksum mismatch: {entry["file"]}'
    assert PROTOCOL['calibration_scene'] not in PROTOCOL['evaluation_scenes']
    hardware=host_metadata()
    (OUT/'environment.json').write_text(json.dumps(hardware,indent=2))
    statuses=[]
    for variant in PROTOCOL['variants']:
        print('Running',variant,flush=True)
        try:
            if variant not in ['rife49-fp32','rife49-half','rife426-fp32']:
                with (OUT/f'{variant}-build.log').open('w') as log:
                    result=subprocess.run([sys.executable,__file__,'--build',variant],stdout=log,stderr=log,timeout=240,env={**os.environ,'OMP_NUM_THREADS':'2'})
                if result.returncode:raise RuntimeError(f'Quantization build failed ({result.returncode}); see build log')
            with (OUT/f'{variant}-run.log').open('w') as log:
                result=subprocess.run([sys.executable,__file__,'--worker',variant],stdout=log,stderr=log,timeout=180)
            if result.returncode:raise RuntimeError(f'CPU inference failed ({result.returncode}); see run log')
            data=json.loads((OUT/f'{variant}.json').read_text());statuses.append(dict(variant=variant,status='complete'))
        except (RuntimeError,subprocess.TimeoutExpired) as e:
            data=dict(variant=variant,status='failed',error=str(e));(OUT/f'{variant}.json').write_text(json.dumps(data,indent=2));statuses.append(data)
        print(statuses[-1],flush=True)
    base=json.loads((OUT/'rife49-fp32.json').read_text());teacher=np.load(OUT/'rife49-fp32-predictions.npy');rows=[];summary=[]
    for status in statuses:
        if status['status']!='complete':continue
        variant=status['variant'];data=json.loads((OUT/f'{variant}.json').read_text());outputs=np.load(OUT/f'{variant}-predictions.npy')
        assert outputs.shape==teacher.shape
        for i,row in enumerate(data['rows']):
            expected=base['rows'][i]
            assert all(row[k]==expected[k] for k in ['scene','left','right','target','t'])
            row['teacher_output_mse']=float(np.mean((outputs[i].astype(float)-teacher[i].astype(float))**2));rows.append(row)
        for scene in PROTOCOL['evaluation_scenes']:
            subset=[r for r in data['rows'] if r['scene']==scene];baseline=[r for r in base['rows'] if r['scene']==scene]
            latency=float(np.median([r['median_seconds'] for r in subset]));base_latency=float(np.median([r['median_seconds'] for r in baseline]))
            delta=float(np.mean([r['psnr'] for r in subset])-np.mean([r['psnr'] for r in baseline]))
            summary.append(dict(variant=variant,scene=scene,n=len(subset),mean_psnr=float(np.mean([r['psnr'] for r in subset])),mean_ssim=float(np.mean([r['ssim'] for r in subset])),delta_psnr_vs_fp32=delta,median_pair_seconds=latency,speedup_vs_fp32=base_latency/latency,model_file_bytes=data['model_file_bytes'],process_peak_rss_mib=data['process_peak_rss_mib'],pilot_gate_passed=bool(base_latency/latency>=1.25 and delta>=-.2)))
    report=dict(protocol=PROTOCOL,protocol_sha256=sha(ROOT/'research/protocols/compression-pilot.json'),environment=hardware,statuses=statuses,summary=summary,rows=rows)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    with (OUT/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
