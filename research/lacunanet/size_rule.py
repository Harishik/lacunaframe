"""Protocol v4 section 5: pick the largest width whose ORT CPU latency <= RIFE 4.9 at 640x360."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from backend.learned import ASSETS, MANIFEST


def session(path):
    o = ort.SessionOptions(); o.intra_op_num_threads = 2; o.inter_op_num_threads = 1
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=o, providers=['CPUExecutionProvider'])


def median_latency(run, n=20):
    for _ in range(3): run()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); run(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


rng = np.random.default_rng(0)
rife = session(ASSETS / MANIFEST['model']['file'])
img = lambda h, w: rng.random((1, 3, h, w), dtype=np.float32)
rife_feed = {'img0': img(384, 640), 'img1': img(384, 640), 'timestep': np.array([.5], np.float32)}
result = dict(rife49=median_latency(lambda: rife.run(None, rife_feed)), widths={})
for c in (16, 24, 32, 48):
    s = session(Path(sys.argv[1]) / f'lacunanet_c{c}.onnx')
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 360
    feed = {k: img(h, 640) for k in ('a_m1', 'a_p1', 'a_m2', 'a_p2')}
    feed['cond'] = np.array([[.5, 0, -1.5, -.5, .5, 1.5, 1, 1]], np.float32)
    result['widths'][c] = median_latency(lambda: s.run(None, feed))
    print(c, result['widths'][c], 'rife', result['rife49'], flush=True)
ok = [c for c, v in result['widths'].items() if v <= result['rife49']]
result['chosen'] = max(ok) if ok else min(result['widths'])
result['rule'] = 'largest C with median ORT latency (2 threads, warm, 20 runs) <= RIFE 4.9 at 640x360 (RIFE padded to 640x384 as in the adapter)'
(REPO / 'research/results/lacunanet').mkdir(parents=True, exist_ok=True)
(REPO / f"research/results/lacunanet/size-rule{sys.argv[3] if len(sys.argv) > 3 else ''}.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
