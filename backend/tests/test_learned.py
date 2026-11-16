import cv2
import numpy as np
from backend.learned import ASSETS, interpolate, model_info, session
from backend.core import restore


def real_frames():
    cap=cv2.VideoCapture(str(ASSETS/'video/blackswan.mp4'));frames=[]
    for _ in range(5):
        ok,a=cap.read();assert ok
        frames.append(cv2.cvtColor(cv2.resize(a,(96,54)),cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def test_pinned_cpu_model_and_nonmultiple_shape():
    assert model_info()['available']
    assert session().get_providers()==['CPUExecutionProvider']
    a=real_frames()
    for fast in (False,True):
        out=interpolate(a[0],a[4],.25,fast)
        assert out.shape==a[0].shape and out.dtype==np.uint8
        assert np.array_equal(out,interpolate(a[0],a[4],.25,fast))
        assert np.isfinite(out).all()
    assert np.array_equal(interpolate(a[0],a[4],0),a[0])
    assert np.array_equal(interpolate(a[0],a[4],1),a[4])


def test_real_pixel_learned_recovery_no_reference_leakage():
    reference=real_frames();missing=np.array([False,False,True,False,False])
    damaged=reference.copy();damaged[missing]=0
    mask=np.zeros(reference.shape[:3],bool)
    full,warnings,_=restore(damaged,missing,mask,'rife')
    assert np.array_equal(full[~missing],reference[~missing])
    assert full[missing].mean()>10
    assert any('Pretrained' in warning for warning in warnings)
    linear,_,_=restore(damaged,missing,mask,'linear')
    assert not np.array_equal(full[missing],linear[missing])
    # A different storage placeholder must not affect the inferred frame.
    damaged[missing]=255
    other,_,_=restore(damaged,missing,mask,'rife')
    assert np.array_equal(full,other)
