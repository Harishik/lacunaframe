import io
import time
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from backend.app import app


def test_complete_path():
    with TestClient(app) as c:
        demos=c.get('/api/demos').json();assert len(demos)==6 and not demos[0]['synthetic']
        s=c.post('/api/scenarios',json={'source':demos[0]['id'],'drop':.1,'gap':2}).json()
        assert len(s['missing'])==s['count']
        assert c.get(f"/api/frames/{s['id']}/corrupted/20").content[:8]==b'\x89PNG\r\n\x1a\n'
        run=c.post('/api/run',json={'scenario':s['id'],'methods':['linear','rife']});assert run.status_code==200
        ident=run.json()['id']
        for _ in range(200):
            j=c.get('/api/results/'+ident).json()
            if j['status']!='running':break
            time.sleep(.1)
        assert j['status']=='complete'
        for m in ('linear','rife'):
            metric=j['report']['methods'][m]
            assert len(metric['frames'])==s['count']
            assert len(c.get(f"/api/download/{ident}/{metric['media']}").content)>100
        assert c.get(f'/api/download/{ident}/report.json').json()['version']=='0.3.0'
        assert b'affected_psnr' in c.get(f'/api/download/{ident}/metrics.csv').content
        assert c.get(f'/api/frames/{ident}/error/10?method=linear').status_code==200
        assert c.get(f'/api/frames/{ident}/original/999').status_code==404
        assert c.get('/api/frames/not-valid/original/0').status_code==404
        assert c.post('/api/scenarios',json={'source':'demo0','drop':2}).status_code==422


def test_upload_validation_and_sequence():
    with TestClient(app) as c:
        assert c.post('/api/upload',files={'files':('a.exe',b'bad','text/plain')}).status_code==400
        assert c.post('/api/upload',files={'files':('a.mp4',b'bad','video/mp4')}).status_code==400
        stream=io.BytesIO();Image.fromarray(np.zeros((32,32,3),np.uint8)).save(stream,format='PNG')
        r=c.post('/api/upload',files=[('files',(f'{i}.png',stream.getvalue(),'image/png')) for i in range(3)])
        assert r.status_code==200 and r.json()['count']==3


def test_real_mp4_upload():
    import backend.app as module
    with TestClient(app) as c:
        media=module.ASSETS/'video/blackswan.mp4'
        r=c.post('/api/upload',files={'files':('sample.mp4',media.read_bytes(),'video/mp4')})
        assert r.status_code==200
        assert r.json()['count']==25 and r.json()['fps']==12
        assert c.get(f"/api/frames/{r.json()['id']}/original/24").status_code==200


def test_gif_fallback_and_cleanup(monkeypatch):
    import os
    import backend.app as module
    folder=module.ROOT/('a'*32)
    folder.mkdir()
    def broken(*args,**kwargs):raise RuntimeError('Codec unavailable')
    monkeypatch.setattr(module.imageio,'mimwrite',broken)
    name=module.encode(folder,'hold',np.zeros((2,8,8,3),np.uint8),12)
    assert name=='hold.gif' and (folder/name).stat().st_size>0
    os.utime(folder,(time.time()-4000,time.time()-4000))
    module.cleanup()
    assert not folder.exists()


def test_missing_model_fails_explicitly(monkeypatch):
    import backend.app as module
    with TestClient(app) as c:
        s=c.post('/api/scenarios',json={'source':'natural-blackswan'}).json()
        monkeypatch.setattr(module,'model_info',lambda:dict(available=False))
        r=c.post('/api/run',json={'scenario':s['id'],'methods':['rife']})
        assert r.status_code==503 and 'weights unavailable' in r.json()['detail']
