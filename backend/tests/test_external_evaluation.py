import numpy as np
import pytest
from scripts.benchmark_pack import score


def test_external_alignment_and_corrupted_support(tmp_path):
    ref=np.full((3,8,8,3),100,np.uint8);affected=np.zeros((3,8,8),bool);affected[1]=True
    times=np.arange(3,dtype=np.float64)/12
    targets=tmp_path/'targets.npz';prediction=tmp_path/'prediction.npz'
    np.savez(targets,frames=ref,affected=affected,timestamps=times,fps=12)
    out=ref.copy();out[1]=90
    np.savez(prediction,frames=out,timestamps=times)
    result=score(targets,prediction,'external-test')['metrics']
    assert result['affected_psnr']==pytest.approx(10*np.log10(255**2/100))
    assert result['psnr']>result['affected_psnr']
    np.savez(prediction,frames=out,timestamps=times+1/12)
    with pytest.raises(ValueError,match='Timestamp mismatch'):score(targets,prediction,'shifted')
    np.savez(prediction,frames=out.astype(float)/255,timestamps=times)
    with pytest.raises(ValueError,match='uint8'):score(targets,prediction,'wrong-range')
