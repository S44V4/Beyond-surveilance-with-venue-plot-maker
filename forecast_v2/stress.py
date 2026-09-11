"""Frozen development perturbations, not a claim of real-camera robustness."""
from pathlib import Path
import numpy as np
import torch
from .common import read,write,sha,verify_data,sources
from .data import Windows,corrected
from .train import batch_of,make_model


@torch.no_grad()
def stress(data,run,device_name=None):
    data,run=Path(data),Path(run);verify_data(data)
    done=read(run/'COMPLETE.json')
    if sha(run/'best.pt')!=done['best_sha256']:raise ValueError('Checkpoint fingerprint mismatch')
    ckpt=torch.load(run/'best.pt',map_location='cpu',weights_only=False);c=ckpt['config'];cal=ckpt['calibration']
    if read(run/'contract.json')['sources']!=sources() or read(run/'contract.json')['manifest_sha256']!=sha(data/'manifest.json'):
        raise ValueError('Restore frozen sources and data before stress evaluation')
    device=torch.device(device_name or ('cuda' if torch.cuda.is_available() else 'cpu'));torch.set_num_threads(2)
    m=make_model(c,cal,ckpt['input_dim']).to(device);m.load_state_dict(ckpt['model']);m.eval()
    ds=Windows(data,'val',c['context'],c['horizons'],1);results={}
    # These are fixed input-level tests. They do not simulate all effects of missed heads on the backbone.
    for condition in ('clean','appearance_missing','motion_missing','last_4_frames_stale','count_scale_0.8','count_scale_1.2'):
        groups={}
        for start in range(0,len(ds),c['micro_batch']):
            b=batch_of(ds,range(start,min(start+c['micro_batch'],len(ds))),'cpu')
            if condition=='appearance_missing':b['x'][...,:256]=0
            elif condition=='motion_missing':b['x'][...,258:]=0
            elif condition=='last_4_frames_stale':
                for k in ('x','anchor','observed'):b[k][:,-4:]=b[k][:,-5:-4].clone()
            elif condition.startswith('count_scale_'):
                factor=float(condition.split('_')[-1]);b['observed'][...,0]*=factor
                b['anchor']=torch.from_numpy(corrected(b['observed'].numpy(),cal))
                for idx in (256,257):
                    raw=b['x'][...,idx]*cal['x_std'][idx]+cal['x_mean'][idx]
                    b['x'][...,idx]=(raw*factor-cal['x_mean'][idx])/cal['x_std'][idx]
            out=m(b['x'].to(device),b['anchor'].to(device),b['observed'].to(device))
            y=b['target'][...,0].numpy();pred=out['count'].cpu().numpy();base=np.repeat(b['anchor'][:,-1,None,:].numpy(),len(c['horizons']),1)
            for i,sid in enumerate(b['scene']):
                groups.setdefault(sid,[]).append([np.abs(pred[i]-y[i]).mean(),np.abs(pred[i].sum(-1)-y[i].sum(-1)).mean(),np.abs(base[i]-y[i]).mean(),np.abs(base[i].sum(-1)-y[i].sum(-1)).mean()])
        means=np.array([np.mean(v,axis=0) for v in groups.values()]).mean(0)
        results[condition]=dict(zip(('model_zone_mae','model_total_mae','calibrated_persistence_zone_mae','calibrated_persistence_total_mae'),map(float,means)))
    write(run/'stress-development.json',{'role':'post-selection development stress diagnostic; no real-world robustness claim','conditions':results,'best_sha256':done['best_sha256']})
    print(results)
