"""Separate CPU process per ORT configuration; real held-out natural frames."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if len(sys.argv)==1:
    rows=[]
    for threads,opt in [(1,'disabled'),(1,'all'),(2,'all'),(4,'all')]:
        result=subprocess.run([sys.executable,__file__,str(threads),opt],capture_output=True,text=True,check=True)
        rows.append(json.loads(result.stdout))
    (ROOT/'examples/natural-v02/cpu-profile.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))
else:
    import time
    import resource
    import cv2
    import numpy as np
    import onnxruntime as ort
    threads=int(sys.argv[1]);opt=sys.argv[2]
    options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL if opt=='all' else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    begin=time.perf_counter();session=ort.InferenceSession(str(ROOT/'assets/models/rife49.onnx'),sess_options=options,providers=['CPUExecutionProvider']);load=time.perf_counter()-begin
    cap=cv2.VideoCapture(str(ROOT/'assets/video/bmx-trees.mp4'));frames=[]
    for i in range(5):
        ok,a=cap.read();assert ok
        a=cv2.cvtColor(cv2.resize(a,(320,180)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
        frames.append(np.pad(a.transpose(2,0,1)[None],((0,0),(0,0),(0,12),(0,0))))
    cap.release()
    feeds={'img0':frames[0],'img1':frames[4],'timestep':np.array([.5],np.float32)}
    session.run(None,feeds)
    durations=[]
    for _ in range(7):
        start=time.perf_counter();out=session.run(None,feeds)[0];durations.append(time.perf_counter()-start)
    print(json.dumps(dict(threads=threads,graph_optimization=opt,provider=session.get_providers(),load_seconds=load,median_seconds=float(np.median(durations)),p90_seconds=float(np.percentile(durations,90)),process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,output_mean=float(out.mean()),samples=durations)))
