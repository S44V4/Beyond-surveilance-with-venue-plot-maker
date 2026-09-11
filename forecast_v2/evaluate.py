"""Development diagnostics, paired comparisons and sequence-level interval calibration."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from .common import read, write, sha, verify_data, sources
from .data import Windows, corrected
from .train import make_model, batch_of, DEFAULT


def metrics(y, count, pressure, thresholds):
    delta=count-y[...,0]; total=delta.sum(-1)
    classes=np.searchsorted(thresholds,y[...,1],side='right').reshape(-1)
    pred=np.searchsorted(thresholds,pressure,side='right').reshape(-1)
    cm=np.bincount(classes*4+pred,minlength=16).reshape(4,4)
    denom=cm.sum(0)+cm.sum(1);f1=np.divide(2*cm.diagonal(),denom,out=np.zeros(4),where=denom>0)
    return {'zone_mae':float(np.abs(delta).mean()),'zone_mse':float((delta**2).mean()),'zone_rmse':float(np.sqrt((delta**2).mean())),'total_mae':float(np.abs(total).mean()),'total_mse':float((total**2).mean()),'total_rmse':float(np.sqrt((total**2).mean())),'pressure_mae':float(np.abs(pressure-y[...,1]).mean()),'pressure_macro_f1':float(f1.mean()),'pressure_confusion':cm.tolist()}


def baselines(batch,horizons):
    raw=batch['observed'][:,-1,:,0].numpy();anchor=batch['anchor'].numpy()
    pressure=batch['observed'][:,-1,:,1].numpy()
    # Least-squares slope of corrected observations, never manual current counts.
    t=np.arange(anchor.shape[1],dtype=float);t-=t.mean()
    slope=np.einsum('t,btn->bn',t,anchor)/(t@t)
    trend=np.maximum(anchor[:,-1,None,:]+np.array(horizons)[None,:,None]*slope[:,None,:],0)
    return {'persistence':np.repeat(raw[:,None,:],len(horizons),1),'calibrated_persistence':np.repeat(anchor[:,-1,None,:],len(horizons),1),'linear_trend':trend},np.repeat(pressure[:,None,:],len(horizons),1)


def diagnose(data,out):
    verify_data(data);dataset=Windows(data,'val',32,[5,15,30],1)
    grouped={};zero=[];changes=[]
    for start in range(0,len(dataset),128):
        b=batch_of(dataset,range(start,min(start+128,len(dataset))),'cpu');pred,p=baselines(b,[5,15,30]);y=b['target'].numpy()
        for i,sid in enumerate(b['scene']):
            d=grouped.setdefault(sid,{'y':[],'p':[],**{k:[] for k in pred}})
            d['y'].append(y[i]);d['p'].append(p[i])
            for k in pred:d[k].append(pred[k][i])
        zero.extend(np.abs(b['observed'][:,-1,:,0].sum(-1).numpy()-b['current'][...,0].sum(-1).numpy()).tolist())
        changes.extend(np.abs(y[...,0].sum(-1)-b['current'][...,0].sum(-1).numpy()[:,None]).mean(-1).tolist())
    rows=[]
    for sid,d in grouped.items():
        d={k:np.array(v) for k,v in d.items()}
        for k in ('persistence','calibrated_persistence','linear_trend'):
            for hi,h in enumerate((5,15,30)):
                rows.append({'scene':sid,'model':k,'horizon':h,**metrics(d['y'][:,hi],d[k][:,hi],d['p'][:,hi],dataset.cal['pressure_thresholds'])})
    summary={k:{m:float(np.mean([r[m] for r in rows if r['model']==k])) for m in ('zone_mae','total_mae')} for k in ('persistence','calibrated_persistence','linear_trend')}
    write(out,{'role':'development validation, not independent evidence','summary':summary,'current_ddpf_total_mae_window_weighted':float(np.mean(zero)),'true_total_temporal_change_mae_window_weighted':float(np.mean(changes)),'rows':rows,'calibration_fitted_on':'train only'})
    print(summary)


@torch.no_grad()
def predictions(model,dataset,config,device):
    grouped={};model.eval()
    for start in range(0,len(dataset),config['micro_batch']):
        b=batch_of(dataset,range(start,min(start+config['micro_batch'],len(dataset))),'cpu')
        baseline,pressure=baselines(b,config['horizons'])
        out=model(b['x'].to(device),b['anchor'].to(device),b['observed'].to(device))
        for i,sid in enumerate(b['scene']):
            d=grouped.setdefault(sid,{'truth':[],'pressure_baseline':[],'frames':[],**{k:[] for k in ('count','pressure','lower','upper','balance_residual')},**{k:[] for k in baseline}})
            d['truth'].append(b['target'][i].numpy());d['pressure_baseline'].append(pressure[i]);d['frames'].append(int(b['frame'][i]))
            for k in baseline:d[k].append(baseline[k][i])
            for k in ('count','pressure','lower','upper','balance_residual'):d[k].append(out[k][i].cpu().numpy())
    return {sid:{k:np.asarray(v) for k,v in d.items()} for sid,d in grouped.items()}


def evaluate(data,run,device_name=None):
    data,run=Path(data),Path(run);manifest=verify_data(data);done=read(run/'COMPLETE.json')
    if sha(run/'best.pt')!=done['best_sha256']:raise ValueError('Selected checkpoint changed')
    payload=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    if read(run/'contract.json')['sources']!=sources():raise ValueError('Restore frozen model sources before evaluation')
    if sha(run/'contract.json')!=payload['contract_sha256'] or read(run/'contract.json')['manifest_sha256']!=sha(data/'manifest.json'):
        raise ValueError('Evaluation contract mismatch')
    config=payload['config'];cal=payload['calibration'];torch.set_num_threads(2)
    device=torch.device(device_name or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model=make_model(config,cal,payload['input_dim']).to(device);model.load_state_dict(payload['model'])
    cal_data=Windows(data,'calibration',config['context'],config['horizons'],1)
    groups=predictions(model,cal_data,config,device)
    scores=[float(np.maximum.reduce([d['lower']-d['truth'][...,0],d['truth'][...,0]-d['upper'],np.zeros_like(d['lower'])]).max()) for d in groups.values()]
    # Maximum error per sequence: all overlapping windows are one calibration unit.
    rank=math.ceil((len(scores)+1)*.8)
    if rank>len(scores):raise ValueError('Insufficient calibration sequences for a finite 80% interval')
    radius=sorted(scores)[rank-1]
    outdir=run/'development-evaluation';outdir.mkdir(exist_ok=True)
    write(outdir/'interval-calibration.json',{'nominal_coverage':.8,'unit':'maximum over all windows, horizons and regions within each sequence','radius':radius,'sequences':list(groups),'scores':scores,'best_sha256':done['best_sha256'],'assumption':'exchangeable independent sequences; physical-location grouping not verified, therefore no unconditional coverage guarantee'})
    val=Windows(data,'val',config['context'],config['horizons'],1);groups=predictions(model,val,config,device);rows=[];failures=[]
    for sid,d in groups.items():
        d['calibrated_lower']=np.maximum(d['lower']-radius,0);d['calibrated_upper']=d['upper']+radius
        np.savez_compressed(outdir/f'{sid}.npz',**d)
        for hi,h in enumerate(config['horizons']):
            for k in ('persistence','calibrated_persistence','linear_trend','count'):
                rows.append({'scene':sid,'model':config['mode'] if k=='count' else k,'horizon':h,**metrics(d['truth'][:,hi],d[k][:,hi],d['pressure'][:,hi] if k=='count' else d['pressure_baseline'][:,hi],cal['pressure_thresholds'])})
            err=np.abs(d['count'][:,hi].sum(-1)-d['truth'][:,hi,:,0].sum(-1))
            for i in np.argsort(err)[-5:]:failures.append({'scene':sid,'frame':int(d['frames'][i]),'horizon':h,'absolute_total_error':float(err[i])})
    comparisons=[];rng=np.random.default_rng(20260909);sids=sorted(groups)
    for baseline in ('persistence','calibrated_persistence','linear_trend'):
        for h in config['horizons']:
            for metric in ('zone_mae','total_mae'):
                values=np.array([next(r[metric] for r in rows if r['scene']==s and r['horizon']==h and r['model']==config['mode'])-next(r[metric] for r in rows if r['scene']==s and r['horizon']==h and r['model']==baseline) for s in sids])
                boot=values[rng.integers(0,len(values),(2000,len(values)))].mean(-1)
                comparisons.append({'baseline':baseline,'horizon':h,'metric':metric,'delta':float(values.mean()),'descriptive_ci95':np.quantile(boot,[.025,.975]).tolist()})
    coverage=[]
    for sid,d in groups.items():
        within=(d['truth'][...,0]>=d['calibrated_lower'])&(d['truth'][...,0]<=d['calibrated_upper'])
        coverage.append({'scene':sid,'marginal_coverage':float(within.mean()),'whole_sequence_covered':bool(within.all()),'mean_width':float((d['calibrated_upper']-d['calibrated_lower']).mean()),'max_balance_residual':float(np.abs(d['balance_residual']).max())})
    write(outdir/'metrics.json',{'role':'post-selection development validation; NOT independent test or SOTA evidence','rows':rows,'comparisons':comparisons,'coverage':coverage,'failure_cases':sorted(failures,key=lambda x:-x['absolute_total_error']),'independent_test_required':True})
    lines=['# Development evaluation','', '**Not an independent test result.** These validation sequences selected the model. A fresh external protocol and test are required before superiority claims.','', '| Model | Horizon | Zone MAE | Total MAE | Pressure MAE | Pressure macro-F1 |','|---|---:|---:|---:|---:|---:|']
    for k in ('persistence','calibrated_persistence','linear_trend',config['mode']):
        for h in config['horizons']:
            values=[r for r in rows if r['model']==k and r['horizon']==h]
            means=[np.mean([r[m] for r in values]) for m in ('zone_mae','total_mae','pressure_mae','pressure_macro_f1')]
            lines.append(f'| {k} | {h} | '+ ' | '.join(f'{v:.4f}' for v in means)+' |')
    (outdir/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Saved development report: {outdir / "REPORT.md"}')


def doctor(data,config_path):
    config={**DEFAULT,**read(config_path),'mode':'transport','seed':23037};manifest=verify_data(data)
    dataset=Windows(data,'train',config['context'],config['horizons'],config['stride']);torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=make_model(config,dataset.cal,manifest['input_dim']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    b=batch_of(dataset,range(config['micro_batch']),device);started=time.monotonic()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True);out=model(b['x'],b['anchor'],b['observed'])
        from .model import objective
        loss=objective(out,b,dataset.cal['count_scale'],model.thresholds);loss.backward();optimizer.step()
    if device.type=='cuda':torch.cuda.synchronize()
    result={'device':str(device),'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,'parameters':sum(p.numel() for p in model.parameters()),'micro_batch':config['micro_batch'],'fp32_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,'seconds_per_microbatch':(time.monotonic()-started)/3,'note':'Three-step engineering memory check. Not research training. Peak allocation excludes driver and other applications.'}
    print(__import__('json').dumps(result,indent=2));return result
