"""Derive plots and an evidence table from the saved measurements only."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'research/results/compression-pilot'


def main() -> None:
    data=json.loads((OUT/'summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,5),facecolor='#eeeae2')
    names={'rife49-fp32':'4.9 FP32 ensemble','rife49-half':'4.9 half resolution','rife426-fp32':'4.26 FP32','rife49-dynamic-u8':'4.9 dynamic U8','rife49-qdq-all-conv':'4.9 QDQ Conv','rife49-qdq-trunk':'4.9 QDQ trunk'}
    colors=['#252a33','#8393a2','#c87553','#4b82ac','#9a547d','#88713e']
    for ax,scene in zip(axes,data['protocol']['evaluation_scenes']):
        ax.set_facecolor('#eeeae2')
        for (variant,label),color in zip(names.items(),colors):
            row=next(r for r in data['summary'] if r['scene']==scene and r['variant']==variant)
            ax.scatter(row['median_pair_seconds']*1000,row['mean_psnr'],s=60+row['model_file_bytes']/2**20*6,label=label,color=color,edgecolor='white')
        ax.set(title=scene,xlabel='Median pair latency · milliseconds ↓',ylabel='Mean midpoint RGB PSNR · dB ↑');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
    fig.suptitle('Compression feasibility on CPU · exploratory 18-triplet pilot\nMarker area reflects model file size. Calibration: separate black-swan sequence. Not an official paper benchmark.',fontsize=12)
    fig.tight_layout(rect=(0,.15,1,.9));fig.savefig(OUT/'quality-latency.png',dpi=170);plt.close(fig)
    lines=['# CPU compression pilot — measured results','','This is a feasibility screen, not a new learned method or a publication benchmark. All six variants execute actual inference on CPU. No distillation, fine-tuning or structural pruning has been performed.','','![Actual quality/latency tradeoff](quality-latency.png)','','| Variant | Scene | RGB PSNR | Δ dB vs FP32 | Median ms | Speed ratio | File MiB | Pilot gate |','|---|---|---:|---:|---:|---:|---:|---|']
    for r in data['summary']:
        lines.append(f"| {names[r['variant']]} | {r['scene']} | {r['mean_psnr']:.3f} | {r['delta_psnr_vs_fp32']:+.3f} | {r['median_pair_seconds']*1000:.2f} | {r['speedup_vs_fp32']:.2f}× | {r['model_file_bytes']/2**20:.2f} | {'pass' if r['pilot_gate_passed'] else 'fail'} |")
    lines+=['','No explicit ORT `quant_pre_process` pass was applied to the already simplified source graph. Improved preprocessing, fusion or export choices may change these results; the pilot does not establish that PTQ is generally ineffective.','','A ratio below 1 means slower than the FP32 ensemble baseline. The preset gate requires at least 1.25× speed and no more than 0.2 dB mean loss on **each** evaluation scene. Only the 4.26 configuration passes both scenes in this run. This does not overturn the earlier arbitrary-time long-gap study: the task, case selection and timestep sampling differ. There is no new independent learned-architecture baseline here.','','## What was actually changed','','- Dynamic U8: ConvInteger activation/weight computation using ORT dynamic quantization; remaining operators stay float.','- QDQ Conv: S8/S8 per-channel static Conv quantization, with activation calibration on black swan only. ConvTranspose, warps and other operators remain float.','- QDQ trunk: quantizes only Conv nodes containing `/convblock/`; leaves input projections and all other operators float.','- Half resolution: uses the unchanged FP32 checkpoint at lower resolution and upsamples. It is **not** a smaller parameter model.','- Neither quantization nor downsampling is a newly trained student or a pruning result.','','## Scope and reproduction','','Run `.venv/bin/python research/scripts/compression_pilot.py`, then `profile_operators.py` and `summarize_pilot.py` from the repository root. The runner overwrites this pilot directory; preserve a copy for subsequent independent trials. See [protocol](../../protocols/compression-pilot.json), [full JSON](summary.json), [CSV](summary.csv), [environment](environment.json), and [operator diagnostics](operator-profile.json).','','Nine calibration triplets from black swan. Eighteen evaluation triplets from two other scenes; anchor separations 2/4/8 and starts 0/12/24; midpoint only. Same normalized RGB arrays and target timestamps for every method. The source scenes were used in earlier exploratory work, so they are not an untouched research test set. Pretraining overlap is unknown. Two intra-op CPU threads and one inter-op thread; fresh process per variant. Quantizer calibration uses ORT internal CPU defaults.','','Unprofiled latency includes input packing, inference, crop, conversion and any resizing. Two warmup calls and three repetitions per triplet; the table is the median across per-triplet medians. These are screening timings, not a robust latency study or significance test. P95 from only three repeats is descriptive interpolation, not a reliable tail-latency estimate. Peak RSS includes Python, imported libraries, normalized data and outputs; it is not isolated model memory.','','Quantized ONNX files and construction manifests are in `models/`. They retain upstream RIFE MIT attribution from [the original assets](../../../assets/THIRD_PARTY.md). Prediction arrays and figures derived from DAVIS retain CC BY-NC 4.0. Build/run logs are retained. `attempts/` records two fixed ORT-reader/calibration integration errors; those were implementation errors, not model failures.','','The pilot does not measure LPIPS, temporal consistency over complete reconstructed sequences, energy, ARM behavior, native 720p performance or a trained student. These are outstanding research requirements.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    expected={'rife49-fp32':ROOT/'assets/models/rife49.onnx','rife426-fp32':ROOT/'assets/models/rife426.onnx'}
    import hashlib
    provenance=dict(protocol_sha256=data['protocol_sha256'],sources={name:hashlib.sha256((ROOT/f'assets/video/{name}.mp4').read_bytes()).hexdigest() for name in ['blackswan','bmx-trees','breakdance']},checkpoints={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in expected.items()},normalization='backend.app.normalize, every second decoded MP4 frame, at most 48 samples')
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print('Wrote measured table, plot and provenance')

if __name__=='__main__':main()
