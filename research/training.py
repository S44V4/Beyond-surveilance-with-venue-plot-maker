"""Full multi-seed training. Test windows are never opened by this module."""
import json
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from research.protocol import ROOT,STUDY,PROTOCOL,atomic_json,sha
from research.cache import Windows
from crowd_twin.models.strfe import STRFE
from crowd_twin.models.graph_reasoner import GraphRiskReasoner
from crowd_twin.venue import VenueGraph,node_static_features

INPUTS=['visual_features','density','localization_logits','flow','zone_masks','sensors','sensor_mask']
HANDOFF=['current_zone_features','future_zone_features','current_zone_state','future_zone_state']

def temporal():return STRFE(256,64,PROTOCOL['horizons'],sensor_features=(),dropout=.1,use_sensors=False)
def graph():return GraphRiskReasoner(64,64,5,4,2,4,PROTOCOL['horizons'],dropout=.1)
def topology(device):
    venue=VenueGraph.load(ROOT/'configs/venues/dronecrowd_analysis_grid.json')
    return torch.tensor(venue.adjacency,dtype=torch.bool,device=device),torch.tensor(node_static_features(venue,feature_dim=5),dtype=torch.float32,device=device)

def inputs(batch,device):return {k:batch[k].to(device) for k in INPUTS}
def graph_forward(model,batch,device,adj,static):
    return model(current_zone_features=batch['current_zone_features'].to(device),future_zone_features=batch['future_zone_features'].to(device),current_state_prior=batch['current_zone_state'].to(device),future_state_prior=batch['future_zone_state'].to(device),adjacency=adj,node_static=static)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def save_torch(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)

def train_one(kind,seed,device):
    folder=STUDY/f'seed-{seed}'/kind
    if (folder/'complete.json').exists():
        if json.loads((folder/'complete.json').read_text())['best_checkpoint_sha256']!=sha(folder/'best.pt'):raise ValueError('Completed checkpoint fingerprint changed')
        return
    folder.mkdir(parents=True,exist_ok=True);seed_all(seed)
    datasets={s:Windows(s) if kind=='strfe' else HandoffWindows(seed,s) for s in ['train','val']}
    model=(temporal() if kind=='strfe' else graph()).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3,factor=.5)
    calibration=json.loads((STUDY/'target-calibration.json').read_text())
    scale=torch.tensor(calibration['state_scale'],device=device,dtype=torch.float32)
    thresholds=torch.tensor(calibration['pressure_thresholds'],device=device,dtype=torch.float32)
    adj,static=topology(device)
    best=float('inf');stale=0;start=0;history=[]
    last=folder/'last.pt'
    if last.exists():
        checkpoint=torch.load(last,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model_state']);optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']);best=checkpoint['best'];stale=checkpoint['stale'];start=checkpoint['epoch']+1;history=checkpoint['history']
        torch.set_rng_state(checkpoint['rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    for epoch in range(start,PROTOCOL[f'{kind}_epochs']):
        if stale>=PROTOCOL['patience'] and epoch>=PROTOCOL['min_epochs']:break
        losses=[];scene_values={};started=time.monotonic()
        for split in ['train','val']:
            training=split=='train';model.train(training)
            loader=DataLoader(datasets[split],batch_size=PROTOCOL['batch_size'],shuffle=training,num_workers=0,generator=torch.Generator().manual_seed(seed*1000+epoch),pin_memory=device.type=='cuda')
            with torch.set_grad_enabled(training):
                for batch_index,batch in enumerate(loader):
                    current=batch['current_target'].to(device);future=batch['future_target'].to(device)
                    if training:optimizer.zero_grad(set_to_none=True)
                    out=model(**inputs(batch,device)) if kind=='strfe' else graph_forward(model,batch,device,adj,static)
                    state_loss=F.smooth_l1_loss(out['current_zone_state']/scale,current/scale)+F.smooth_l1_loss(out['future_zone_state']/scale,future/scale)
                    loss=state_loss
                    if kind=='graph':
                        cl=torch.bucketize(current[...,6].contiguous(),thresholds);fl=torch.bucketize(future[...,6].contiguous(),thresholds)
                        loss=loss+.25*(F.cross_entropy(out['risk_logits'].reshape(-1,4),cl.reshape(-1))+F.cross_entropy(out['future_risk_logits'].reshape(-1,4),fl.reshape(-1)))+.1*F.binary_cross_entropy(out['hazard_probability'].clamp(1e-6,1-1e-6),(cl>=2).float())
                    if not torch.isfinite(loss):raise ValueError(f'Non-finite {kind} loss')
                    if training:
                        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        if not torch.isfinite(norm):raise ValueError('Non-finite gradients')
                        optimizer.step();losses.append(float(loss.detach()))
                    else:
                        errors=(out['future_zone_state']-future).abs()
                        score=(errors[...,0].mean((1,2))/scale[0]+errors[...,6].mean((1,2)))/2
                        for sid,value in zip(batch['scene'],score.cpu().tolist()):scene_values.setdefault(sid,[]).append(value)
                    if batch_index%100==0:
                        atomic_json(STUDY/'status.json',{'stage':f'train-{kind}','seed':seed,'epoch':epoch+1,'split':split,'batch':batch_index,'batches':len(loader),'updated_unix':time.time()})
        validation=float(np.mean([np.mean(v) for v in scene_values.values()]))
        scheduler.step(validation)
        improved=validation<best
        if improved:best=validation;stale=0
        else:stale+=1
        history.append({'epoch':epoch+1,'training_loss':float(np.mean(losses)),'validation_score':validation,'seconds':time.monotonic()-started,'lr':optimizer.param_groups[0]['lr']})
        checkpoint={'kind':kind,'seed':seed,'epoch':epoch,'model_state':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'best':best,'stale':stale,'history':history,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'protocol_sha256':sha(STUDY/'protocol.json'),'manifest_sha256':sha(STUDY/'split-manifest.json'),'ddpf_sha256':json.loads((STUDY/'split-manifest.json').read_text())['ddpf_sha256'],'deployment_approved':False}
        if kind=='graph':checkpoint['source_strfe_sha256']=sha(STUDY/f'seed-{seed}'/'strfe/best.pt')
        if improved:save_torch(folder/'best.pt',checkpoint)
        save_torch(last,checkpoint);atomic_json(folder/'history.json',history)
        print(json.dumps({'kind':kind,'seed':seed,**history[-1]}),flush=True)
    atomic_json(folder/'complete.json',{'best_checkpoint_sha256':sha(folder/'best.pt'),'epochs':len(history),'best_validation_score':best,'early_stopped':len(history)<PROTOCOL[f'{kind}_epochs']})

@torch.inference_mode()
def export_handoffs(seed,device):
    model=temporal().to(device)
    path=STUDY/f'seed-{seed}'/'strfe/best.pt';fingerprint=sha(path)
    model.load_state_dict(torch.load(path,map_location=device,weights_only=False)['model_state']);model.eval()
    for split in ['train','val']:
        folder=STUDY/f'seed-{seed}'/'handoff'/split;complete=folder/'complete.json'
        if complete.exists():
            if json.loads(complete.read_text())['source_strfe_sha256']!=fingerprint:raise ValueError('Stale handoffs')
            continue
        folder.mkdir(parents=True,exist_ok=True);dataset=Windows(split);outputs={};offset=0
        for batch in DataLoader(dataset,batch_size=2,num_workers=0):
            out=model(**inputs(batch,device));n=len(batch['scene'])
            for key in HANDOFF:
                value=out[key].cpu().numpy()
                if key not in outputs:outputs[key]=np.lib.format.open_memmap(folder/f'{key}.npy',mode='w+',dtype=np.float32,shape=(len(dataset),*value.shape[1:]))
                outputs[key][offset:offset+n]=value
            offset+=n
            if offset%200==0:atomic_json(STUDY/'status.json',{'stage':'graph-handoff','seed':seed,'split':split,'windows':offset,'total_windows':len(dataset),'updated_unix':time.time()})
        for value in outputs.values():value.flush()
        del outputs
        atomic_json(folder/'index.json',dataset.index)
        atomic_json(complete,{'source_strfe_sha256':fingerprint,'windows':offset})

class HandoffWindows(Dataset):
    def __init__(self,seed,split):
        if split not in ['train','val']:raise ValueError('Test handoffs are generated only during sealed evaluation')
        folder=STUDY/f'seed-{seed}'/'handoff'/split
        self.index=json.loads((folder/'index.json').read_text());self.outputs={k:np.load(folder/f'{k}.npy',mmap_mode='r',allow_pickle=False) for k in HANDOFF};self.targets={}
    def __len__(self):return len(self.index)
    def __getitem__(self,i):
        sid,t=self.index[i]
        if sid not in self.targets:self.targets[sid]=np.load(STUDY/'cache'/sid/'target.npy',mmap_mode='r',allow_pickle=False)
        states=self.targets[sid]
        value={k:torch.from_numpy(v[i].copy()) for k,v in self.outputs.items()}
        value.update(scene=sid,frame=t,current_target=torch.from_numpy(states[t].copy()),future_target=torch.from_numpy(states[[t+h for h in PROTOCOL['horizons']]].copy()))
        return value

def train_all():
    torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for seed in PROTOCOL['seeds']:
        train_one('strfe',seed,device);export_handoffs(seed,device);train_one('graph',seed,device)
    sealed={f'{seed}/{kind}':sha(STUDY/f'seed-{seed}'/kind/'best.pt') for seed in PROTOCOL['seeds'] for kind in ['strfe','graph']}
    atomic_json(STUDY/'CHECKPOINTS_SEALED.json',{'checkpoints':sealed,'protocol_sha256':sha(STUDY/'protocol.json'),'manifest_sha256':sha(STUDY/'split-manifest.json'),'sealed_unix':time.time()})
