"""Protect exported-model grid layout; candidate remains an opt-in research path."""
import numpy as np
import pytest
from backend.candidate import pack,interpolate,INFO
from backend.learned import ASSETS


def test_candidate_coordinate_contract():
    a=np.zeros((54,96,3),np.uint8);b=np.full_like(a,255)
    value=pack(a,b,.25)
    assert value.shape==(1,11,64,128) and value.dtype==np.float32
    assert np.all(value[0,:3]==0) and np.all(value[0,3:6,:54,:96]==1)
    assert np.all(value[0,6]==.25)
    assert value[0,7,0,0]==-1 and value[0,7,0,-1]==1
    assert value[0,8,0,0]==-1 and value[0,8,-1,0]==1
    np.testing.assert_allclose(value[0,9],2/127)
    np.testing.assert_allclose(value[0,10],2/63)
    with pytest.raises(ValueError):pack(a,b,1.1)


@pytest.mark.skipif(not (ASSETS/INFO['file']).exists(),reason='Optional candidate checkpoint not installed')
def test_candidate_cpu_output_and_endpoints():
    from backend.candidate import session
    from backend.tests.test_learned import real_frames
    a=real_frames()
    assert session().get_providers()==['CPUExecutionProvider']
    out=interpolate(a[0],a[4],.5)
    assert out.shape==a[0].shape and out.dtype==np.uint8 and out.mean()>10
    np.testing.assert_array_equal(interpolate(a[0],a[4],0),a[0])
    np.testing.assert_array_equal(interpolate(a[0],a[4],1),a[4])
