"""Exercise upload, persistent GPU processing, snapshots, scenarios and export."""
import json
import sys
import time
from pathlib import Path

import cv2
import httpx

root = Path(__file__).resolve().parents[1]
source = Path(sys.argv[1])
out = root / 'artifacts' / 'verification-video.mp4'
images = sorted(source.glob('img011*.jpg'))[:4]
if not images:
    raise SystemExit('No verification images found')
writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), 1, (960, 540))
for path in images:
    writer.write(cv2.resize(cv2.imread(str(path)), (960, 540)))
writer.release()
client = httpx.Client(base_url='http://127.0.0.1:8010/api', timeout=60)
with out.open('rb') as handle:
    response = client.post('/videos', files={'file': (out.name, handle, 'video/mp4')})
response.raise_for_status()
video = response.json()
response = client.post('/runs', json={'video_id': video['id'], 'venue_id': 'atrium-demo', 'sample_fps': 1, 'camera_name': 'Verification camera', 'idempotency_key': f'verification-{time.time_ns()}'})
if response.status_code == 404:
    venue = client.get('/venues').json()[-1]['id']
    response = client.post('/runs', json={'video_id': video['id'], 'venue_id': venue, 'sample_fps': 1, 'camera_name': 'Verification camera', 'idempotency_key': f'verification-{time.time_ns()}'})
response.raise_for_status()
job = response.json()
deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    job = client.get('/runs/' + job['id']).json()
    print(job['status'], job.get('progress'), flush=True)
    if job['status'] in ['completed', 'failed', 'cancelled']:
        break
    time.sleep(5)
assert job['status'] == 'completed', job
timeline = client.get(f"/runs/{job['id']}/timeline").json()
export = client.get(f"/runs/{job['id']}/export")
export.raise_for_status()
report = {'run_id': job['id'], 'status': job['status'], 'result': job.get('result'), 'timeline': timeline, 'export_bytes': len(export.content), 'source': str(source), 'scope': 'Four real dataset images encoded as a test video; transport and inference verification, not an accuracy evaluation.'}
(root / 'artifacts' / 'video-verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({'run_id': job['id'], 'status': job['status'], 'export_bytes': len(export.content)}))
