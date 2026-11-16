"""Checks that compression evidence comes from aligned actual pixels and CPU graphs."""
from collections import Counter
import json
from pathlib import Path
import hashlib
import numpy as np
import onnx
import onnxruntime as ort
from backend.core import evaluate
from research.scripts.compression_pilot import PROTOCOL,OUT,feeds,frames,pairs,path_for


def test_scene_split_and_target_alignment():
    assert PROTOCOL['calibration_scene'] not in PROTOCOL['evaluation_scenes']
    for scene in [PROTOCOL['calibration_scene'],*PROTOCOL['evaluation_scenes']]:
        source=frames(scene)
        starts=PROTOCOL['calibration_starts'] if scene==PROTOCOL['calibration_scene'] else PROTOCOL['evaluation_starts']
        for case in pairs(scene,starts):
            assert case['target']==(case['left']+case['right'])//2
            np.testing.assert_array_equal(case['reference'],source[case['target']])
            packed=feeds(case['a'],case['b'],case['t'],'rife49-fp32')
            np.testing.assert_array_equal(packed['img0'][0,:,:180,:320].transpose(1,2,0),case['a'].astype(np.float32)/255)
            assert set(packed)=={'img0','img1','timestep'}


def test_saved_scores_are_recomputed_from_actual_predictions():
    data=json.loads((OUT/'summary.json').read_text())
    sources={s:frames(s) for s in PROTOCOL['evaluation_scenes']}
    for status in data['statuses']:
        assert status['status']=='complete'
        variant=status['variant'];prediction=np.load(OUT/f'{variant}-predictions.npy',allow_pickle=False)
        rows=[r for r in data['rows'] if r['variant']==variant]
        assert prediction.shape==(len(rows),180,320,3) and prediction.dtype==np.uint8
        for i,row in enumerate(rows):
            ref=sources[row['scene']][row['target']][None]
            actual=evaluate(ref,prediction[i:i+1],np.ones((1,180,320),bool),12)
            assert abs(actual['psnr']-row['psnr'])<1e-9
            assert abs(actual['ssim']-row['ssim'])<1e-9


def test_quantized_graphs_execute_on_cpu_and_match_saved_output():
    case=pairs('bmx-trees',PROTOCOL['evaluation_starts'])[0]
    for variant in ['rife49-dynamic-u8','rife49-qdq-all-conv','rife49-qdq-trunk']:
        path=path_for(variant);metadata=json.loads(path.with_suffix('.json').read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==metadata['output_sha256']
        ops=Counter(n.op_type for n in onnx.load(path).graph.node)
        assert ops['ConvInteger']>0 if 'dynamic' in variant else ops['QuantizeLinear']>0
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        assert session.get_providers()==['CPUExecutionProvider']
        value=session.run(None,feeds(case['a'],case['b'],.5,variant))[0]
        assert np.isfinite(value).all()
        image=np.clip(value[0,:,:180,:320].transpose(1,2,0)*255,0,255).round().astype(np.uint8)
        expected=np.load(OUT/f'{variant}-predictions.npy',allow_pickle=False)[0]
        np.testing.assert_array_equal(image,expected)


def test_calibration_source_and_operator_profile_provenance():
    for variant in ['rife49-qdq-all-conv','rife49-qdq-trunk']:
        metadata=json.loads(path_for(variant).with_suffix('.json').read_text())
        assert {p['scene'] for p in metadata['calibration_pairs']}=={PROTOCOL['calibration_scene']}
        assert len(metadata['calibration_pairs'])==9
    for profile in json.loads((OUT/'operator-profile.json').read_text()):
        assert (OUT/profile['profile_file']).is_file()
        assert profile['operators'] and abs(sum(p['share'] for p in profile['operators'])-1)<1e-9
