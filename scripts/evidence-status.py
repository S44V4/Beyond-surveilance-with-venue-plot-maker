"""Read-only study status with a rolling rate that is robust to long pauses."""
import json
import statistics
import sys
import time
from pathlib import Path

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from research.run import alive

folder=root/'outputs/evidence-20260907'
status=json.loads((folder/'status.json').read_text())
lock=json.loads((folder/'run.lock').read_text()) if (folder/'run.lock').exists() else None
result={'stage':status['stage'],'process_alive':bool(lock and alive(lock['pid'])),'pid':lock['pid'] if lock else None,'seconds_since_progress':round(time.time()-status['updated_unix'],1),'completed':(folder/'EVALUATION_COMPLETE.json').exists()}
if status['stage']=='extract':
    log=folder/'run.stdout.log'
    records=[]
    if log.exists():
        with log.open('rb') as file:
            file.seek(max(0,log.stat().st_size-100000))
            lines=file.read().decode('utf-8',errors='replace').splitlines()
        for line in lines:
            try:
                item=json.loads(line)
                if item.get('stage')=='extract':records.append(item)
            except (ValueError,TypeError):pass
    rates=[]
    for a,b in zip(records[-9:],records[-8:]):
        frames=b['completed_frames']-a['completed_frames']
        if frames>0:rates.append((b['updated_unix']-a['updated_unix'])/frames)
    result.update(frames=status['completed_frames'],total_frames=status['total_frames'],percent=round(status['completed_frames']/status['total_frames']*100,2),sequence=status['sequence'])
    if rates:
        rate=statistics.median(rates)
        result.update(recent_seconds_per_frame=round(rate,2),estimated_active_extraction_hours_remaining=round((status['total_frames']-status['completed_frames'])*rate/3600,1),estimate_note='Rolling median interval rate; excludes the effect of a single long pause. Training/evaluation time is additional.')
else:result['details']=status
if (folder/'failure.json').exists():result['last_failure']=json.loads((folder/'failure.json').read_text())
print(json.dumps(result,indent=2))
