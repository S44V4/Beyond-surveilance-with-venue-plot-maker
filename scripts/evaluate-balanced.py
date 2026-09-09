"""Compare compatible graph forecasts with observed DDPF persistence on held-out scenes."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from dataclasses import replace
from crowd_twin.data.strfe import CachedDDPFSequence
from crowd_twin.strfe_inference import STRFEGraphPredictor

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('experiment',type=Path)
args=parser.parse_args()
torch.set_num_threads(2)
folder=args.experiment.resolve()
predictor=STRFEGraphPredictor(folder/'config.yaml',folder/'strfe/strfe_best.pt',folder/'graph/best.pt',device='cpu')
manifest=json.loads((folder/'cache/manifest.json').read_text())
rows=[]
for record in manifest['records']:
    if record['split']!='test': continue
    path=Path(record['path'])
    if not path.is_absolute():path=folder/'cache'/path
    cache=CachedDDPFSequence.load(path)
    for t in range(predictor.context_frames-1,len(cache.density)-max(predictor.strfe.horizons)):
        start=t-predictor.context_frames+1
        window=slice(start,t+1)
        sequence=CachedDDPFSequence(replace(cache.metadata,frame_indices=cache.metadata.frame_indices[window],timestamps_ms=cache.metadata.timestamps_ms[window]),cache.visual_features[window],cache.density[window],cache.localization_logits[window],cache.flow[window],cache.zone_masks)
        prediction=predictor.predict_sequence(sequence)
        observed=(cache.density[t,0][None]*cache.zone_masks).sum((1,2))
        for h,forecast in zip(predictor.strfe.horizons,prediction['forecasts']):
            values=np.array([z['count'] for z in forecast['zones']])
            truth=cache.zone_state[t+h,:,0]
            rows.append({'scene':cache.metadata.sequence_id,'frame':t,'horizon_steps':h,'model_zone_mae':float(np.abs(values-truth).mean()),'persistence_zone_mae':float(np.abs(observed-truth).mean()),'model_total_error':float(abs(values.sum()-truth.sum())),'persistence_total_error':float(abs(observed.sum()-truth.sum()))})
if not rows: raise ValueError('No complete held-out windows')
metrics={key:float(np.mean([row[key] for row in rows])) for key in ['model_zone_mae','persistence_zone_mae','model_total_error','persistence_total_error']}
report={'metrics':metrics,'rows':rows,'test_scenes':sorted({r['scene'] for r in rows}),'strfe_sha256':predictor.strfe_sha256,'graph_sha256':predictor.graph.checkpoint_sha256,'engineering_smoke':predictor.strfe_checkpoint.get('engineering_smoke',False),'baseline':'Current DDPF estimates carried forward; no current or future manual counts are provided to the baseline.','units':'count errors in people; horizons in normalized frame steps','deployment_approved':False}
(folder/'heldout-comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
