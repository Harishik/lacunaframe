"""Diagnostic CPU operator timing; not a substitute for unprofiled latency."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import onnxruntime as ort
from research.scripts.compression_pilot import OUT,PROTOCOL,feeds,frames,path_for


def main() -> None:
    a=frames('bmx-trees');reports=[]
    for variant in ['rife49-fp32','rife49-dynamic-u8','rife49-qdq-all-conv']:
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_profiling=True;options.profile_file_prefix=str(OUT/f'profile-{variant}');options.log_severity_level=3
        session=ort.InferenceSession(str(path_for(variant)),sess_options=options,providers=['CPUExecutionProvider'])
        for left,right in [(0,2),(0,4),(0,8)]:session.run(None,feeds(a[left],a[right],.5,variant))
        profile=Path(session.end_profiling())
        totals=defaultdict(float);counts=defaultdict(int)
        events=json.loads(profile.read_text())
        for event in events:
            if event.get('cat')=='Node' and event.get('name','').endswith('_kernel_time'):
                op=event.get('args',{}).get('op_name','unknown');totals[op]+=event['dur'];counts[op]+=1
        denominator=sum(totals.values())
        reports.append(dict(variant=variant,profile_file=profile.name,model_sha256=hashlib.sha256(path_for(variant).read_bytes()).hexdigest(),scope='Three full calls, including first-call effects. Share of summed Node kernel event durations, not end-to-end latency or FLOPs. Instrumentation changes timing.',operators=[dict(op=op,duration_us=duration,events=counts[op],share=duration/denominator) for op,duration in sorted(totals.items(),key=lambda item:-item[1])]))
        print(variant,reports[-1]['operators'][:4],flush=True)
    (OUT/'operator-profile.json').write_text(json.dumps(reports,indent=2))

if __name__=='__main__':main()
