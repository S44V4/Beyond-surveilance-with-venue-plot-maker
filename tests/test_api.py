import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient
from backend import store
from backend.api import app
from backend.worker import process_scenario


@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DATA',tmp_path)
    monkeypatch.setenv('CROWD_NO_WORKER','1')
    with TestClient(app) as client:
        yield client


def test_venue_version_conflict_and_saved_run_immutability(client):
    venue=client.get('/api/venues').json()[0]
    saved=client.post('/api/venues',json={**venue,'name':'Updated venue'})
    assert saved.status_code==200
    assert saved.json()['version']==venue['version']+1
    assert client.post('/api/venues',json=venue).status_code==409
    run=client.post('/api/examples').json()
    snapshot_venue=run['body']['venue']
    current=next(v for v in client.get('/api/venues').json() if v['id']=='atrium-study')
    assert client.post('/api/venues',json={**current,'name':'Changed after run'}).status_code==200
    assert client.get(f"/api/runs/{run['id']}").json()['body']['venue']==snapshot_venue


def test_complete_scenario_and_export(client):
    run=client.post('/api/examples').json()
    rid=run['id']
    assert client.get(f'/api/runs/{rid}/timeline').json()['total']==1
    payload={'run_id':rid,'snapshot_index':0,'idempotency_key':'scenario-test-123','duration':30,'gates':{'E-N':{'open':False}}}
    response=client.post('/api/scenarios',json=payload)
    assert response.status_code==200,response.text
    job=response.json()
    assert client.post('/api/scenarios',json=payload).json()['id']==job['id']
    process_scenario(job);store.update_job(job['id'],'completed',1)
    result=client.get(f"/api/scenarios/{job['id']}").json()['result']
    assert result['scenario']['metrics']['gate_totals']['E-N']==0
    assert result['baseline']['metrics']['gate_totals']['E-N']>0
    exported=client.get(f'/api/runs/{rid}/export')
    assert exported.status_code==200
    with zipfile.ZipFile(io.BytesIO(exported.content)) as z:
        assert 'timeline.jsonl' in z.namelist()
        assert json.loads(z.read('run.json'))['id']==rid


def test_invalid_media_and_invalid_scenario(client):
    assert client.post('/api/videos',files={'file':('bad.mp4',b'not a video','video/mp4')}).status_code==422
    assert client.post('/api/videos',files={'file':('bad.txt',b'hello','text/plain')}).status_code==422
    run=client.post('/api/examples').json()
    base={'run_id':run['id'],'snapshot_index':0,'idempotency_key':'invalid-gate-test'}
    assert client.post('/api/scenarios',json={**base,'gates':{'missing':{'open':False}}}).status_code==422
    assert client.post('/api/scenarios',json={**base,'snapshot_index':999}).status_code==422
    assert client.post('/api/scenarios',json={**base,'start':200,'duration':30}).status_code==422


def test_cancel_retry_and_idempotency(client):
    run=client.post('/api/examples').json()
    payload={'run_id':run['id'],'snapshot_index':0,'idempotency_key':'cancel-test-key'}
    job=client.post('/api/scenarios',json=payload).json()
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()['status']=='cancelled'
    retried=client.post(f"/api/jobs/{job['id']}/retry").json()
    assert retried['id']!=job['id']
    assert retried['status']=='queued'
    assert client.post('/api/scenarios',json={**payload,'duration':50}).status_code==409


def test_external_origin_and_asset_traversal(client):
    assert client.post('/api/examples',headers={'Origin':'https://external.example'}).status_code==403
    assert client.get('/api/assets/not-a-real-asset').status_code==404
    assert client.get('/api/runs/missing/frames/0.jpg').status_code==404


def test_unobserved_zone_requires_assumption(client):
    run=client.post('/api/examples').json()
    snap=store.frame(run['id'],0);snap['zones'][0]['count']=None;store.add_frame(run['id'],0,snap)
    request={'run_id':run['id'],'snapshot_index':0,'idempotency_key':'missing-initial'}
    assert client.post('/api/scenarios',json=request).status_code==422
    assert client.post('/api/scenarios',json={**request,'unobserved_counts':{'Z01':10}}).status_code==200

