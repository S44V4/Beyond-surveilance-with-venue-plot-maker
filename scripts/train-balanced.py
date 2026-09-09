"""Rebuild frozen features and train compatible STRFE/graph candidates.

Count targets are manual DroneCrowd annotations. Other states remain explicitly
perception-derived proxies. Candidate artifacts are never silently deployed.
"""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
import cv2
import numpy as np
import torch
import yaml
from scipy.io import loadmat
from backend.balanced_ddpf import BalancedPredictor
from crowd_twin.data.strfe import CachedDDPFMetadata, CachedDDPFSequence, optical_flow_sequence, fit_risk_proxy_calibration, zone_states_from_perception, write_strfe_manifest, audit_strfe_manifest
from crowd_twin.strfe_training import train_strfe
from crowd_twin.venue import VenueGraph

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset',type=Path,required=True)
parser.add_argument('--smoke',action='store_true',help='Three disjoint sequences, 12 frames each, two epochs: engineering check only')
parser.add_argument('--epochs',type=int,default=20)
args=parser.parse_args()
torch.set_num_threads(2)
output=ROOT/'outputs'/('balanced-smoke' if args.smoke else 'balanced-training')
output.mkdir(parents=True,exist_ok=True)
venue=VenueGraph.load(ROOT/'configs/venues/dronecrowd_analysis_grid.json')
# Hard partition without overlapping raster edges, shared with serving below.
yy,xx=np.indices((32,32))
cell=np.minimum(yy*3//32,2)*3+np.minimum(xx*3//32,2)
masks=np.stack([(cell==i).astype(np.float32) for i in range(9)])
predictor=BalancedPredictor()
ddpf_hash=predictor.sha256
caches=[]
labels={}
seen=set()
# The local val_data duplicates official test scenes. Deduplicate by sequence
# and create a recorded deterministic scene split before generating windows.
all_groups={}
for path in sorted(args.dataset.rglob('img*.jpg')):
    all_groups.setdefault(path.stem[3:6],{}).setdefault(path.name,path)
ids=sorted(all_groups,key=lambda sid:hashlib.sha256(f'23037:{sid}'.encode()).hexdigest())
if len(ids)<3: raise ValueError('Need at least three distinct source sequences')
a=max(1,int(len(ids)*.6)); b=min(len(ids)-1,max(a+1,int(len(ids)*.8)))
partitions={'train':ids[:a],'val':ids[a:b],'test':ids[b:]}
(output/'scene-split.json').write_text(json.dumps({'method':'seeded sequence-disjoint 60/20/20 custom split; duplicated local val/test copies removed','partitions':partitions},indent=2))
for split,selected_ids in partitions.items():
    groups={sid:sorted(all_groups[sid].values()) for sid in selected_ids}
    for sid,paths in list(groups.items())[:1 if args.smoke else None]:
        if sid in seen:
            raise ValueError(f'Sequence {sid} appears in multiple splits')
        seen.add(sid)
        paths=paths[:12] if args.smoke else paths
        destination=output/'cache'/split/f'{sid}.npz'
        targetfile=destination.with_suffix('.counts.npy')
        if destination.exists() and targetfile.exists():
            cache=CachedDDPFSequence.load(destination)
            if cache.metadata.augmentation_provenance.get('ddpf_sha256')!=ddpf_hash:
                raise ValueError('Cached checkpoint fingerprint changed; use a new output folder')
            labels[sid]=np.load(targetfile,allow_pickle=False)
            caches.append((cache,destination))
            continue
        features=[]; densities=[]; logits=[]; flows=[]; truth=[]; previous=None
        for index,path in enumerate(paths):
            bgr=cv2.imread(str(path))
            if bgr is None: raise ValueError(f'Unreadable {path}')
            sample=predictor.export_sequence([bgr],sid,1)
            features.append(sample.visual_features[0]); densities.append(sample.density[0]); logits.append(sample.localization_logits[0])
            rgb=cv2.cvtColor(cv2.resize(bgr,(512,512)),cv2.COLOR_BGR2RGB)
            flows.append(optical_flow_sequence(np.stack([previous,rgb]),(32,32))[1] if previous is not None else np.zeros((4,32,32),np.float32))
            previous=rgb
            annotation=path.parent.parent/'ground_truth'/f'GT_{path.stem}.mat'
            info=loadmat(annotation,simplify_cells=True)['image_info']
            points=np.asarray(info['location'],dtype=float).reshape(-1,3)
            col=np.clip((points[:,0]/bgr.shape[1]*3).astype(int),0,2)
            row=np.clip((points[:,1]/bgr.shape[0]*3).astype(int),0,2)
            truth.append(np.bincount(row*3+col,minlength=9).astype(np.float32))
            print(f'{split} sequence {sid}: {index+1}/{len(paths)}',flush=True)
        metadata=CachedDDPFMetadata(schema_version='1.0',sequence_id=sid,source_sequence=sid,split=split,venue_id=venue.venue_id,zone_ids=venue.zone_ids,fps=1,frame_indices=tuple(range(len(paths))),timestamps_ms=tuple(i*1000 for i in range(len(paths))),feature_channels=256,source='DroneCrowd_balanced_DDPF_manual_counts',augmentation_provenance={'ddpf_sha256':ddpf_hash,'preprocessing':'valid-image-features-v1','physical_fps_available':False})
        cache=CachedDDPFSequence(metadata,np.stack(features),np.stack(densities),np.stack(logits),np.stack(flows),masks)
        cache.save(destination)
        labels[sid]=np.stack(truth)
        np.save(targetfile,labels[sid])
        caches.append((cache,destination))
del predictor
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
calibration=fit_risk_proxy_calibration([c for c,p in caches if c.metadata.split=='train'],scale_quantile=.95,component_weights={'density':1.,'motion':1.,'convergence':1.})
calibration_path=output/'risk_calibration.json'
calibration_path.write_text(json.dumps(calibration,indent=2))
for cache,path in caches:
    cache.zone_state=zone_states_from_perception(cache.density,cache.flow,cache.zone_masks,calibration)
    cache.zone_state[:,:,0]=labels[cache.metadata.sequence_id]
    cache.zone_state[:,:,1]=labels[cache.metadata.sequence_id]/np.maximum(masks.sum((1,2)),1)
    cache.save(path)
manifest=write_strfe_manifest([p for c,p in caches],output/'cache'/'manifest.json')
audit=audit_strfe_manifest(manifest)
assert audit['clean'],audit
config=yaml.safe_load((ROOT/'configs/strfe_dronecrowd_research.yaml').read_text())
config['output_dir']=str(output/'strfe')
config['data'].update(manifest=str(manifest),risk_calibration=str(calibration_path),context_frames=4 if args.smoke else 8)
config['model'].update(input_channels=256,horizons_seconds=[1,2,4] if args.smoke else [5,15,30])
config['training'].update(epochs=2 if args.smoke else args.epochs,batch_size=2,mixed_precision=False,device='cpu')
config['ddpf_sha256']=ddpf_hash
config['feature_preprocessing']='valid-image-features-v1'
config_path=output/'config.yaml'
config_path.write_text(yaml.safe_dump(config))
result=train_strfe(config)
checkpoint=torch.load(result['checkpoint'],map_location='cpu',weights_only=False)
checkpoint.update(ddpf_sha256=ddpf_hash,feature_preprocessing='valid-image-features-v1',deployment_approved=False,count_supervision='manual DroneCrowd head locations',engineering_smoke=args.smoke)
torch.save(checkpoint,result['checkpoint'])
env={**os.environ,'PYTHONPATH':str(ROOT)}
subprocess.run([sys.executable,'scripts/train_graph_reasoner_dataset.py','--config',str(config_path),'--strfe-checkpoint',str(result['checkpoint']),'--output',str(output/'graph'/'best.pt'),'--epochs',str(2 if args.smoke else args.epochs),'--batch-size','2','--device','cpu'],check=True,env=env)
summary={'ddpf_sha256':ddpf_hash,'strfe':result,'leakage_audit':audit,'engineering_smoke':args.smoke,'deployment_approved':False,'scope':'Count supervision uses manual head annotations. Pressure uses derived ordinal proxies. Physical timing and causal interventions are not validated. Smoke mode verifies training execution only.'}
(output/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
