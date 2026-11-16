import numpy as np
import cv2
import pytest
from backend.core import demo, corrupt, restore, motion, flow_between, evaluate


def cfg(**kw):
    return dict(seed=42,drop=0,gap=1,start=.5,occlusion=0,noise=0,blur=0,**kw)


def test_linear_and_boundaries():
    a=np.stack([np.full((8,8,3),i*20,np.uint8) for i in range(5)])
    missing=np.array([True,False,True,False,True]); damaged=a.copy();damaged[missing]=0
    out,w,_=restore(damaged,missing,np.zeros((5,8,8),bool),'linear')
    assert np.all(out[2]==40)
    assert np.array_equal(out[0],a[1]) and np.array_equal(out[-1],a[-2])
    assert any('Boundary' in x for x in w)
    assert np.array_equal(out[~missing],a[~missing])


def test_no_anchor():
    with pytest.raises(ValueError): restore(np.zeros((2,8,8,3),np.uint8),np.ones(2,bool),np.zeros((2,8,8),bool),'flow')


def test_flow_direction_translation():
    rng=np.random.default_rng(3)
    a=rng.integers(0,255,(64,96,3),dtype=np.uint8)
    b=cv2.warpAffine(a,np.float32([[1,0,4],[0,1,0]]),(96,64))
    f,r=flow_between(a,b)
    assert f.shape==(64,96,2) and f.dtype==np.float32
    assert np.median(f[12:-12,12:-12,0])==pytest.approx(4,abs=.5)
    assert np.median(r[12:-12,12:-12,0])==pytest.approx(-4,abs=.5)
    actual,u=motion(a,b,.5)
    truth=cv2.warpAffine(a,np.float32([[1,0,2],[0,1,0]]),(96,64))
    assert np.abs(actual[12:-12,12:-12].astype(float)-truth[12:-12,12:-12]).mean()<8
    assert 0<=u<=1


def test_corruption_reproducible_alignment():
    a=demo(1,n=10,h=80,w=128); c=cfg();c.update(drop=.3,occlusion=.2,noise=5)
    d,m,k=corrupt(a,c);d2,m2,k2=corrupt(a,c)
    assert np.array_equal(d,d2) and np.array_equal(m,m2) and np.array_equal(k,k2)
    assert d.shape==a.shape and m.shape==(10,) and k.shape==a.shape[:3]
    assert np.all(d[m]==0) and np.any(a[m]!=0)


def test_metrics_affected_and_small():
    a=np.zeros((3,8,8,3),np.uint8);b=a.copy();b[1,0,0]=255
    mask=np.zeros((3,8,8),bool);mask[1,0,0]=True
    result=evaluate(a,b,mask,12)
    assert result['affected_psnr']==0
    assert result['psnr']>20
    assert result['frames'][0]['psnr']=='Infinity'
    assert result['frames'][1]['time']==1/12
    assert result['temporal_error']>0
    identical=evaluate(a,a,mask,12)
    assert identical['ssim']==1 and identical['psnr']=='Infinity'
    assert evaluate(a[:,:2,:2],b[:,:2,:2],mask[:,:2,:2],12)['ssim'] is None
    with pytest.raises(ValueError): evaluate(a,np.full(a.shape,np.nan),mask,12)


def test_telea_only_changes_masks():
    a=np.full((3,24,24,3),100,np.uint8);mask=np.zeros((3,24,24),bool);mask[:,8:16,8:16]=True
    damaged=a.copy();damaged[mask]=0
    out,_,_=restore(damaged,np.zeros(3,bool),mask,'telea')
    assert np.array_equal(out[~mask],a[~mask])
    assert np.abs(out[mask].astype(float)-100).mean()<5


def test_evaluation_rejects_invalid_reference_and_timestamps():
    ref=np.zeros((2,4,4,3),np.float32);mask=np.zeros((2,4,4),bool)
    bad=ref.copy();bad[0,0,0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):evaluate(bad,ref,mask,12)
    with pytest.raises(ValueError,match='FPS'):evaluate(ref,ref,mask,0)
    with pytest.raises(ValueError,match='aligned'):evaluate(ref,ref,mask[:1],12)
