"""Component ablation of a trained v3 (saved by cpu_sanity): switch learned parts off one at a time
by zeroing their ReZero gates, and report dev PSNR. Dev only.

    ../.venv-train/bin/python research/lacunanet/ablate.py <state.pt>
"""
import copy, json, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch
from research.lacunanet.model_v2 import LacunaNetV2
from research.lacunanet.cpu_sanity import evaluate


def main():
    os.environ.setdefault('SANITY_CLIPS', '6'); torch.set_num_threads(6)
    st = torch.load(sys.argv[1], map_location='cpu')
    base = LacunaNetV2(24, prior='lk'); base.load_state_dict(st['model'])
    gains = {n: p.detach().flatten().tolist() for n, p in base.named_parameters() if n.endswith('gain')}
    print('gates', json.dumps({k: [round(x, 4) for x in v] for k, v in gains.items()}), flush=True)
    variants = {
        'trained': [],
        'no_pair_residual': ['pair.heads.0.gain', 'pair.heads.1.gain', 'pair.heads.2.gain'],
        'no_correction_flow': ['cor_gain:0-4'],
        'no_correction_mask': ['cor_gain:4-5'],
        'no_refinement': ['ref_gain'],
        'prior_only': ['pair.heads.0.gain', 'pair.heads.1.gain', 'pair.heads.2.gain', 'cor_gain:0-5', 'ref_gain'],
    }
    out = dict(gates=gains)
    for name, zero in variants.items():
        m = copy.deepcopy(base)
        with torch.no_grad():
            for z in zero:
                if ':' in z:
                    pn, rng = z.split(':'); a, b = map(int, rng.split('-'))
                    dict(m.named_parameters())[pn][a:b] = 0
                else:
                    dict(m.named_parameters())[z].zero_()
        out[name] = evaluate(m)
        print(name, {g: (v['model'], v.get('warp_blend_only')) for g, v in out[name].items()}, flush=True)
    (REPO / 'research/results/lacunanet/ablation-gated-v3.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
