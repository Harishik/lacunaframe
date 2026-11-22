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
def paired_latency(run_a, run_b, n=20):
    """Interleaved timing (a, b, a, b, ...) so both see the same background load. Returns medians and the
    median of per-pair ratios b/a, which is robust to load drifting during the measurement."""
    for _ in range(3): run_a(); run_b()
    ta, tb = [], []
    for _ in range(n):
        t = time.perf_counter(); run_a(); ta.append(time.perf_counter() - t)
        t = time.perf_counter(); run_b(); tb.append(time.perf_counter() - t)
    return float(np.median(ta)), float(np.median(tb)), float(np.median(np.array(tb) / np.array(ta)))


rife_run = lambda: rife.run(None, rife_feed)
result = dict(rife49=median_latency(rife_run), widths={}, ratio_to_rife={})
for c in (16, 24, 32, 48):
    s = session(Path(sys.argv[1]) / f'lacunanet_c{c}.onnx')
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 360
    feed = {k: img(h, 640) for k in ('a_m1', 'a_p1', 'a_m2', 'a_p2')}
    feed['cond'] = np.array([[.5, 0, -1.5, -.5, .5, 1.5, 1, 1]], np.float32)
    r_med, m_med, ratio = paired_latency(rife_run, lambda: s.run(None, feed))
    result['widths'][c], result['ratio_to_rife'][c] = m_med, ratio
    print(c, m_med, 'rife', r_med, 'ratio', round(ratio, 3), flush=True)
ok = [c for c, v in result['ratio_to_rife'].items() if v <= 1.0]
result['chosen'] = max(ok) if ok else min(result['widths'])
result['rule'] = 'largest C whose median interleaved latency ratio to RIFE 4.9 is <= 1 (ORT, 2 threads, warm, 20 pairs, 640x360; RIFE padded to 640x384 as in the adapter)'
result['load_average'] = __import__('os').getloadavg()
(REPO / 'research/results/lacunanet').mkdir(parents=True, exist_ok=True)
(REPO / f"research/results/lacunanet/size-rule{sys.argv[3] if len(sys.argv) > 3 else ''}.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
