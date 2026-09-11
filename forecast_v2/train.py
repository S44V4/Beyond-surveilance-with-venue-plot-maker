"""Optimizer-boundary checkpoints preserve weights, optimizer, RNG and data cursor."""
import json
import math
import os
from pathlib import Path
import platform
import signal
import time
import traceback
import numpy as np
import torch
from torch.utils.data import default_collate
from .common import read, write, sha, verify_data, sources, lock
from .data import Windows
from .model import RegionalForecaster, objective


DEFAULT = {'epochs': 60, 'min_epochs': 15, 'patience': 10, 'context': 32, 'horizons': [5,15,30], 'stride': 5, 'dim': 64, 'dropout': .1, 'micro_batch': 8, 'accumulation': 4, 'learning_rate': .0003, 'weight_decay': .01, 'checkpoint_steps': 20, 'precision': 'auto', 'threads': 2, 'appearance_dropout': .1}


def save(path, value):
    path = Path(path); temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def make_model(config, cal, input_dim):
    return RegionalForecaster(input_dim, config['dim'], config['horizons'], config['mode'], config['dropout'], cal['count_scale'], cal['pressure_thresholds'])


def batch_of(dataset, indices, device):
    value = default_collate([dataset[int(i)] for i in indices])
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in value.items()}


@torch.no_grad()
def validate(model, dataset, config, device):
    model.eval(); groups = {}
    for start in range(0,len(dataset),config['micro_batch']):
        batch = batch_of(dataset, range(start,min(start+config['micro_batch'],len(dataset))),device)
        out = model(batch['x'], batch['anchor'], batch['observed'])
        zone = (out['count']-batch['target'][...,0]).abs().mean((1,2))
        total = (out['count'].sum(-1)-batch['target'][...,0].sum(-1)).abs().mean(1)
        pressure = (out['pressure']-batch['target'][...,1]).abs().mean((1,2))
        for i,sid in enumerate(batch['scene']):
            groups.setdefault(sid, []).append([zone[i].item(),total[i].item(),pressure[i].item()])
    means = np.array([np.mean(v,axis=0) for v in groups.values()]).mean(0)
    # Count-first selection; pressure remains a separately reported proxy task.
    score = (means[0]+means[1]/3)/dataset.cal['count_scale']
    return {'selection_score':float(score),'zone_count_mae':float(means[0]),'total_count_mae':float(means[1]),'pressure_mae':float(means[2])}


def train(data, run, config, max_steps=None, device_name=None):
    run, data = Path(run),Path(data)
    with lock(run):
        try:
            return _train(data,run,config,max_steps,device_name)
        except Exception as error:
            write(run/'failure.json',{'error':str(error),'traceback':traceback.format_exc(),'updated_unix':time.time()})
            raise


def _train(data,run,config,max_steps,device_name):
    manifest = verify_data(data); config = {**DEFAULT, **config}
    if config['mode'] not in ('transport','direct','state_only'): raise ValueError('Invalid mode')
    if config['min_epochs']>config['epochs'] or min(config['micro_batch'],config['accumulation'],config['checkpoint_steps'])<1:
        raise ValueError('Invalid training budget')
    torch.set_num_threads(config['threads'])
    torch.manual_seed(config['seed']); np.random.seed(config['seed'])
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    device=torch.device(device_name or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    precision=config['precision']
    if precision=='auto': precision='bf16' if device.type=='cuda' and torch.cuda.is_bf16_supported() else 'fp32'
    if precision not in ('bf16','fp16','fp32'): raise ValueError('Unknown precision')
    if precision!='fp32' and device.type!='cuda': raise ValueError('Mixed precision requires CUDA in this runner')
    if precision=='bf16' and not torch.cuda.is_bf16_supported(): raise ValueError('GPU does not support BF16; use fp32')
    contract={'config':config,'manifest_sha256':sha(data/'manifest.json'),'sources':sources(),'resolved_precision':precision,'torch_version':str(torch.__version__),'numpy_version':np.__version__}
    contract_path=run/'contract.json'
    if contract_path.exists():
        if read(contract_path)!=contract: raise ValueError('Resume contract changed. Restore code/config/data or use a new run; never silently continue.')
    else: write(contract_path,contract)
    if (run/'COMPLETE.json').exists(): return read(run/'COMPLETE.json')
    cal=read(data/'calibration.json')
    dataset=Windows(data,'train',config['context'],config['horizons'],config['stride'])
    val=Windows(data,'val',config['context'],config['horizons'],1)
    model=make_model(config,cal,manifest['input_dim']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3,factor=.5)
    scaler=torch.amp.GradScaler('cuda',enabled=precision=='fp16')
    state={'epoch':0,'step_in_epoch':0,'global_step':0,'history':[],'best':float('inf'),'stale':0,'running_loss':0.,'running_steps':0,'epoch_seconds':0.}
    last=run/'last.pt'
    if last.exists():
        payload=torch.load(last,map_location='cpu',weights_only=False)
        if payload['contract_sha256']!=sha(contract_path): raise ValueError('Checkpoint contract mismatch')
        model.load_state_dict(payload['model']); optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler']);scaler.load_state_dict(payload['scaler'])
        state=payload['state'];torch.set_rng_state(payload['rng'])
        if device.type=='cuda' and payload['cuda_rng']: torch.cuda.set_rng_state_all(payload['cuda_rng'])
    environment={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'device':str(device),'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,'precision':precision,'started_unix':time.time()}
    write(run/f'environment-{time.time_ns()}.json',environment)
    def checkpoint():
        save(last,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'state':state,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'contract_sha256':sha(contract_path)})
    pause={'requested':False}
    previous=signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT,lambda *_:pause.update(requested=True))
    (run/'PAUSE').unlink(missing_ok=True)
    executed=0; effective=config['micro_batch']*config['accumulation']
    try:
        while state['epoch']<config['epochs']:
            if state['epoch']>=config['min_epochs'] and state['stale']>=config['patience']:break
            order=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(config['seed']+state['epoch'])).tolist()
            total_steps=math.ceil(len(order)/effective);model.train()
            for step in range(state['step_in_epoch'],total_steps):
                started=time.monotonic(); indices=order[step*effective:(step+1)*effective]
                optimizer.zero_grad(set_to_none=True); value=0.
                for offset in range(0,len(indices),config['micro_batch']):
                    ids=indices[offset:offset+config['micro_batch']];batch=batch_of(dataset,ids,device)
                    # Appearance-only dropout. Count/pressure observations remain available; this is not a missed-head simulation.
                    if config['appearance_dropout']:
                        mask=torch.rand((*batch['x'].shape[:3],1),device=device)<config['appearance_dropout']
                        batch['x'][...,:256]=batch['x'][...,:256].masked_fill(mask,0)
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16 if precision=='bf16' else torch.float16,enabled=precision!='fp32'):
                        out=model(batch['x'],batch['anchor'],batch['observed'])
                    loss=objective(out,batch,cal['count_scale'],model.thresholds)*len(ids)/len(indices)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss; last stable checkpoint preserved')
                    scaler.scale(loss).backward();value+=loss.item()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                scaler.step(optimizer);scaler.update()
                state['step_in_epoch']=step+1;state['global_step']+=1;state['running_loss']+=value;state['running_steps']+=1;state['epoch_seconds']+=time.monotonic()-started;executed+=1
                stopping=pause['requested'] or (run/'PAUSE').exists() or (max_steps is not None and executed>=max_steps)
                if state['global_step']%config['checkpoint_steps']==0 or stopping: checkpoint()
                if step%10==0 or stopping:
                    write(run/'status.json',{'stage':'paused' if stopping else 'training','epoch':state['epoch']+1,'optimizer_step':step+1,'epoch_steps':total_steps,'global_step':state['global_step'],'updated_unix':time.time()})
                if stopping:
                    print('Paused safely. Run the same train command to resume.',flush=True);return {'paused':True}
            started=time.monotonic();metrics=validate(model,val,config,device);state['epoch_seconds']+=time.monotonic()-started
            improved=metrics['selection_score']<state['best']
            state['stale']=0 if improved else state['stale']+1
            if improved:
                state['best']=metrics['selection_score']
                save(run/'best.pt',{'model':model.state_dict(),'config':config,'calibration':cal,'input_dim':manifest['input_dim'],'epoch':state['epoch']+1,'contract_sha256':sha(contract_path),'deployment_approved':False})
            scheduler.step(metrics['selection_score'])
            row={'epoch':state['epoch']+1,**metrics,'training_loss':state['running_loss']/max(state['running_steps'],1),'seconds':state['epoch_seconds'],'peak_gpu_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0}
            state['history'].append(row);print(json.dumps(row),flush=True)
            state.update(epoch=state['epoch']+1,step_in_epoch=0,running_loss=0.,running_steps=0,epoch_seconds=0.)
            checkpoint();write(run/'history.json',state['history'])
            if pause['requested'] or (run/'PAUSE').exists():
                write(run/'status.json',{'stage':'paused','completed_epochs':state['epoch'],'global_step':state['global_step'],'updated_unix':time.time()})
                print('Paused safely after validation. Rerun the same command to resume.',flush=True)
                return {'paused':True}
            if state['epoch']>=config['min_epochs'] and state['stale']>=config['patience']: break
        complete={'epochs':state['epoch'],'best_validation_score':state['best'],'best_sha256':sha(run/'best.pt'),'contract_sha256':sha(contract_path),'role':'development candidate, no independent test result','parameters':sum(p.numel() for p in model.parameters())}
        write(run/'COMPLETE.json',complete);write(run/'status.json',{'stage':'complete','updated_unix':time.time(),**complete});return complete
    finally:
        signal.signal(signal.SIGINT,previous)
