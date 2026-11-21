"""Export LacunaNet to ONNX (run in the training environment)."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.lacunanet.model import LacunaNet, make_cond
from research.lacunanet.model_v2 import LacunaNetV2


def export(model: LacunaNet, path: Path, h: int = 360, w: int = 640) -> None:
    model.eval()
    x = [torch.rand(1, 3, h, w) for _ in range(4)]
    cond = torch.tensor([make_cond(.5, 1, 0., 0., False, False)])
    with torch.no_grad():
        torch.onnx.export(model, (*x, cond), str(path), opset_version=17,
                          input_names=['a_m1', 'a_p1', 'a_m2', 'a_p2', 'cond'], output_names=['frame', 'scale'],
                          dynamic_axes={k: {2: 'h', 3: 'w'} for k in ('a_m1', 'a_p1', 'a_m2', 'a_p2', 'frame', 'scale')})


if __name__ == '__main__':
    if sys.argv[1] in ('grid', 'grid-v2', 'grid-v3'):
        out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
        cls = {'grid-v2': LacunaNetV2, 'grid-v3': lambda c: LacunaNetV2(c, prior='lk')}.get(sys.argv[1], LacunaNet)
        for c in (16, 24, 32, 48):
            torch.manual_seed(0)
            export(cls(c), out / f'lacunanet_c{c}.onnx', 360 if cls is LacunaNet else 368, 640)
            print('exported', c)
    else:
        state = torch.load(sys.argv[1], map_location='cpu')
        v = state.get('version', 1)
        model = LacunaNetV2(state['c'], prior='lk' if v == 3 else None) if v >= 2 else LacunaNet(state['c'])
        model.load_state_dict(state['model'])
        export(model, Path(sys.argv[2]), 368 if v >= 2 else 360, 640)
        print('exported', sys.argv[2])
