"""Restartable full-frame extraction and bounded-memory temporal datasets."""
import json
import time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from research.protocol import ROOT,STUDY,DATA,PROTOCOL,atomic_json,sha,freeze
from crowd_twin.data.strfe import optical_flow_sequence,fit_risk_proxy_calibration,zone_states_from_perception

FIELDS={'visual_features':(256,32,32),'density':(1,32,32),'localization_logits':(1,32,32),'flow':(4,32,32),'counts':(9,)}

def masks():
    yy,xx=np.indices((32,32));cells=np.minimum(yy*3//32,2)*3+np.minimum(xx*3//32,2)
    return np.stack([(cells==i).astype(np.float32) for i in range(9)])

def arrays(sid,mode='r'):
    folder=STUDY/'cache'/sid
    return {k:np.load(folder/f'{k}.npy',mmap_mode=mode,allow_pickle=False) for k in FIELDS}

def extract():
    from backend.balanced_ddpf import BalancedPredictor
    manifest=freeze();torch.set_num_threads(2)
    model=BalancedPredictor()
    if model.sha256!=manifest['ddpf_sha256']:raise ValueError('DDPF fingerprint changed')
    import timm
    atomic_json(STUDY/'runtime.json',{'torch':torch.__version__,'timm':timm.__version__,'cuda_runtime':torch.version.cuda,'device':str(model.device),'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,'ddpf_sha256':model.sha256,'feature_storage_dtype':'float16','compute_dtype':'bfloat16 if supported, otherwise float32','annotation_coordinates':'MATLAB 1-based coordinates converted to image origin before assigning 3x3 cells','feature_grid':'32x32 valid-image-aligned','num_threads':2})
    started=time.monotonic();done_at_start=0;new=0
    for split in ['train','val','test']:
        for sid in manifest['partitions'][split]:
            rows=manifest['sequences'][sid]['frames'];n=len(rows);folder=STUDY/'cache'/sid
            if (folder/'complete.json').exists():
                metadata=json.loads((folder/'complete.json').read_text())
                if metadata['ddpf_sha256']!=model.sha256 or metadata['frames']!=n:raise ValueError('Cache contract mismatch')
                done_at_start+=n;continue
            folder.mkdir(parents=True,exist_ok=True)
            for field,shape in FIELDS.items():
                path=folder/f'{field}.npy'
                if not path.exists():
                    arr=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16 if field=='visual_features' else np.float32,shape=(n,*shape));arr.flush();del arr
            progress_path=folder/'progress.json'
            begin=json.loads(progress_path.read_text())['frames_done'] if progress_path.exists() else 0
            done_at_start+=begin
            values=arrays(sid,'r+');previous=None
            if begin:
                prior=cv2.imread(str(DATA/rows[begin-1]['image']))
                previous=cv2.cvtColor(cv2.resize(prior,(512,512)),cv2.COLOR_BGR2RGB)
            max_quant_error=0.
            for i in range(begin,n):
                row=rows[i];image_path=DATA/row['image'];annotation=DATA/row['annotation']
                if sha(image_path)!=row['image_sha256'] or sha(annotation)!=row['annotation_sha256']:raise ValueError('Source changed after split freeze')
                bgr=cv2.imread(str(image_path))
                if bgr is None:raise ValueError(f'Cannot decode {image_path}')
                sample=model.export_sequence([bgr],sid,1)
                rgb=cv2.cvtColor(cv2.resize(bgr,(512,512)),cv2.COLOR_BGR2RGB)
                flow=optical_flow_sequence(np.stack([previous,rgb]),(32,32))[1] if previous is not None else np.zeros((4,32,32),np.float32)
                previous=rgb
                pts=np.asarray(loadmat(annotation,simplify_cells=True)['image_info']['location'],dtype=float)
                if pts.size:pts=pts.reshape(-1,pts.shape[-1])
                else:pts=np.empty((0,2))
                col=np.clip(((pts[:,0]-1)/bgr.shape[1]*3).astype(int),0,2)
                r=np.clip(((pts[:,1]-1)/bgr.shape[0]*3).astype(int),0,2)
                features=sample.visual_features[0]
                half=features.astype(np.float16)
                if not np.isfinite(half).all():raise ValueError('Feature cache quantization overflow')
                max_quant_error=max(max_quant_error,float(np.abs(half.astype(np.float32)-features).max()))
                values['visual_features'][i]=half
                values['density'][i]=sample.density[0]
                values['localization_logits'][i]=sample.localization_logits[0]
                values['flow'][i]=flow
                values['counts'][i]=np.bincount(r*3+col,minlength=9)
                for array in values.values():array.flush()
                atomic_json(progress_path,{'frames_done':i+1,'ddpf_sha256':model.sha256})
                new+=1;elapsed=time.monotonic()-started
                status={'stage':'extract','sequence':sid,'split':split,'sequence_frame':i+1,'sequence_frames':n,'new_frames':new,'completed_frames':done_at_start+new,'total_frames':manifest['all_unique_frames'],'elapsed_seconds':elapsed,'seconds_per_frame':elapsed/new,'estimated_remaining_seconds':(manifest['all_unique_frames']-done_at_start-new)*elapsed/new,'updated_unix':time.time()}
                atomic_json(STUDY/'status.json',status)
                if i%10==0:print(json.dumps(status),flush=True)
            del values
            atomic_json(folder/'complete.json',{'frames':n,'ddpf_sha256':model.sha256,'split':split,'max_quantization_absolute_error_this_segment':max_quant_error,'cache_sha256':{k:sha(folder/f'{k}.npy') for k in FIELDS}})
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()

def calibrate():
    manifest=freeze();zone_masks=masks()
    sequences=[]
    for sid in manifest['partitions']['train']:
        values=arrays(sid)
        sequences.append(SimpleNamespace(metadata=SimpleNamespace(split='train',source_sequence=sid),density=values['density'],flow=values['flow'],zone_masks=zone_masks))
    calibration=fit_risk_proxy_calibration(sequences,scale_quantile=.95,component_weights=PROTOCOL['proxy_weights'])
    atomic_json(STUDY/'pressure-calibration.json',calibration)
    train_states=[]
    for split,ids in manifest['partitions'].items():
        for sid in ids:
            values=arrays(sid)
            observed=zone_states_from_perception(values['density'],values['flow'],zone_masks,calibration)
            target=observed.copy();target[:,:,0]=values['counts']
            target[:,:,1]=values['counts']/zone_masks.sum((1,2))[None]*100
            np.save(STUDY/'cache'/sid/'observed.npy',observed)
            np.save(STUDY/'cache'/sid/'target.npy',target)
            if split=='train':train_states.append(target.reshape(-1,7))
    states=np.concatenate(train_states)
    scale=np.maximum(np.quantile(np.abs(states),.95,axis=0),1.)
    thresholds=np.quantile(states[:,6],[.25,.5,.75])
    if np.any(np.diff(thresholds)<=0):raise ValueError('Non-distinct training pressure quartiles')
    atomic_json(STUDY/'target-calibration.json',{'state_scale':scale.tolist(),'pressure_thresholds':thresholds.tolist(),'source_sequences':manifest['partitions']['train'],'source_split':'train'})

class Windows(Dataset):
    def __init__(self,split,stride=None):
        self.manifest=freeze();self.split=split;self.maps={};self.zone_masks=torch.from_numpy(masks())
        stride=stride or (PROTOCOL['train_window_stride'] if split=='train' else 1)
        self.index=[(sid,t) for sid in self.manifest['partitions'][split] for t in range(PROTOCOL['context_frames']-1,len(self.manifest['sequences'][sid]['frames'])-max(PROTOCOL['horizons']),stride)]
    def __len__(self):return len(self.index)
    def __getitem__(self,index):
        sid,t=self.index[index]
        if sid not in self.maps:
            self.maps[sid]=arrays(sid)
            self.maps[sid].update({k:np.load(STUDY/'cache'/sid/f'{k}.npy',mmap_mode='r',allow_pickle=False) for k in ['target','observed']})
        values=self.maps[sid];window=slice(t-PROTOCOL['context_frames']+1,t+1)
        data={k:torch.from_numpy(np.array(values[k][window],dtype=np.float32)) for k in ['visual_features','density','localization_logits','flow']}
        data.update(zone_masks=self.zone_masks,sensors=torch.zeros(PROTOCOL['context_frames'],0),sensor_mask=torch.zeros(PROTOCOL['context_frames'],0),current_target=torch.from_numpy(values['target'][t].copy()),future_target=torch.from_numpy(values['target'][[t+h for h in PROTOCOL['horizons']]].copy()),observed=torch.from_numpy(values['observed'][t].copy()),scene=sid,frame=t)
        return data
