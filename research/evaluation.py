"""One sealed test pass, full prediction artifacts, scene/horizon metrics and gate."""
import csv
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from research.protocol import STUDY,PROTOCOL,atomic_json,sha
from research.cache import Windows
from research.training import temporal,graph,inputs,topology,graph_forward
from research.metrics import summarize,cluster_interval,COUNT_METRICS,confusion,pressure_scores

def write_csv(path,rows):
    if not rows:return
    fields=[k for k,v in rows[0].items() if isinstance(v,(str,int,float,bool))]
    with path.open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

@torch.inference_mode()
def evaluate():
    seal=json.loads((STUDY/'CHECKPOINTS_SEALED.json').read_text())
    for key,digest in seal['checkpoints'].items():
        seed,kind=key.split('/')
        if sha(STUDY/f'seed-{seed}'/kind/'best.pt')!=digest:raise ValueError('Checkpoint changed after seal')
    if seal['protocol_sha256']!=sha(STUDY/'protocol.json') or seal['manifest_sha256']!=sha(STUDY/'split-manifest.json'):raise ValueError('Protocol or split changed')
    torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');adj,static=topology(device)
    dataset=Windows('test');cal=json.loads((STUDY/'target-calibration.json').read_text());thresholds=np.array(cal['pressure_thresholds'])
    outdir=STUDY/'evaluation';outdir.mkdir(exist_ok=True)
    atomic_json(outdir/'evaluation_started.json',{'checkpoint_seal_sha256':sha(STUDY/'CHECKPOINTS_SEALED.json'),'started_unix':time.time()})
    metrics=[];failures=[];strata=[]
    for seed in PROTOCOL['seeds']:
        tmodel=temporal().to(device);gmodel=graph().to(device)
        tmodel.load_state_dict(torch.load(STUDY/f'seed-{seed}/strfe/best.pt',map_location=device,weights_only=False)['model_state'])
        gmodel.load_state_dict(torch.load(STUDY/f'seed-{seed}/graph/best.pt',map_location=device,weights_only=False)['model_state'])
        tmodel.eval();gmodel.eval();groups={}
        for batch_index,batch in enumerate(DataLoader(dataset,batch_size=2,num_workers=0)):
            tout=tmodel(**inputs(batch,device));gout=graph_forward(gmodel,tout,device,adj,static)
            truth=batch['future_target'][...,[0,6]].numpy()
            p=np.repeat(batch['observed'][:,None,...,[0,6]].numpy(),len(PROTOCOL['horizons']),axis=1)
            predictions={'persistence':p,'strfe':tout['future_zone_state'][...,[0,6]].cpu().numpy(),'strfe_graph':gout['future_zone_state'][...,[0,6]].cpu().numpy()}
            head=gout['future_risk_logits'].argmax(-1).cpu().numpy()
            current=batch['current_target'][...,0].sum(-1).numpy()
            for i,sid in enumerate(batch['scene']):
                record=groups.setdefault(sid,{'truth':[],'frames':[],'current_true_total':[],'graph_pressure_head':[],**{k:[] for k in predictions}})
                record['truth'].append(truth[i]);record['frames'].append(int(batch['frame'][i]));record['current_true_total'].append(current[i]);record['graph_pressure_head'].append(head[i])
                for k,value in predictions.items():record[k].append(value[i])
            if batch_index%100==0:atomic_json(STUDY/'status.json',{'stage':'heldout-evaluation','seed':seed,'batch':batch_index,'batches':(len(dataset)+1)//2,'updated_unix':time.time()})
        for sid,record in groups.items():
            record={k:np.asarray(v) for k,v in record.items()}
            np.savez_compressed(outdir/f'predictions-{seed}-{sid}.npz',**record)
            for hidx,horizon in enumerate(PROTOCOL['horizons']):
                truth=record['truth'][:,hidx]
                for variant in ['persistence','strfe','strfe_graph']:
                    pred=record[variant][:,hidx]
                    values=summarize(truth,pred,thresholds)
                    if variant=='strfe_graph':
                        head_cm=confusion(np.digitize(truth[...,1],thresholds),record['graph_pressure_head'][:,hidx])
                        values.update(graph_pressure_head_confusion=head_cm.tolist(),**{f'graph_pressure_head_{k}':v for k,v in pressure_scores(head_cm).items()})
                    metrics.append({'seed':seed,'scene':sid,'horizon':horizon,'variant':variant,**values})
                    errors=np.abs(pred[...,0].sum(-1)-truth[...,0].sum(-1))
                    for idx in np.argsort(errors)[-5:]:
                        failures.append({'seed':seed,'scene':sid,'frame':int(record['frames'][idx]),'horizon':horizon,'variant':variant,'true_total':float(truth[idx,:,0].sum()),'predicted_total':float(pred[idx,:,0].sum()),'absolute_total_error':float(errors[idx])})
                    # Prespecified descriptive strata use outcomes only after sealing.
                    counts=truth[...,0].sum(-1);change=np.abs(counts-record['current_true_total'])
                    for name,mask in [('count_lt100',counts<100),('count_100to300',(counts>=100)&(counts<300)),('count_ge300',counts>=300),('change_lt10',change<10),('change_ge10',change>=10)]:
                        if mask.any():strata.append({'seed':seed,'scene':sid,'horizon':horizon,'variant':variant,'stratum':name,**summarize(truth[mask],pred[mask],thresholds)})
        del tmodel,gmodel
    atomic_json(outdir/'scene-horizon-metrics.json',metrics);write_csv(outdir/'scene-horizon-metrics.csv',metrics)
    atomic_json(outdir/'failure-cases.json',sorted(failures,key=lambda r:-r['absolute_total_error']));write_csv(outdir/'failure-cases.csv',sorted(failures,key=lambda r:-r['absolute_total_error']))
    atomic_json(outdir/'strata.json',strata);write_csv(outdir/'strata.csv',strata)
    aggregate(metrics,outdir)

def aggregate(rows,outdir):
    seeds=PROTOCOL['seeds'];scenes=sorted({r['scene'] for r in rows});lookup={(r['seed'],r['scene'],r['horizon'],r['variant']):r for r in rows}
    comparisons=[];gates={};scene_deltas=[]
    for variant in ['strfe','strfe_graph']:
        passing=True
        for horizon in PROTOCOL['horizons']:
            for metric in COUNT_METRICS+['pressure_mae','pressure_rmse','pressure_macro_f1','pressure_balanced_accuracy']:
                key=metric.replace('_rmse','_mse')
                def values(v):return np.array([[lookup[(seed,scene,horizon,v)][key] for scene in scenes] for seed in seeds])
                result=cluster_interval(values(variant),values('persistence'),rmse=metric.endswith('_rmse'),replicates=PROTOCOL['bootstrap_replicates'])
                comparisons.append({'variant':variant,'horizon':horizon,'metric':metric,**result})
                if metric in COUNT_METRICS:passing=passing and result['simultaneous_upper_one_sided95']<0
        seed_checks=[]
        for seed in seeds:
            for metric in ['zone_count_mae','total_count_mae']:
                delta=np.mean([lookup[(seed,s,h,variant)][metric]-lookup[(seed,s,h,'persistence')][metric] for s in scenes for h in PROTOCOL['horizons']])
                seed_checks.append({'seed':seed,'metric':metric,'delta':float(delta)})
        scene_wins=[]
        for scene in scenes:
            deltas={metric:float(np.mean([lookup[(seed,scene,h,variant)][metric]-lookup[(seed,scene,h,'persistence')][metric] for seed in seeds for h in PROTOCOL['horizons']])) for metric in ['zone_count_mae','total_count_mae']}
            scene_deltas.append({'variant':variant,'scene':scene,**deltas});scene_wins.append(all(v<0 for v in deltas.values()))
        pressure_mae=np.mean([r['delta'] for r in comparisons if r['variant']==variant and r['metric']=='pressure_mae'])
        pressure_f1=np.mean([r['delta'] for r in comparisons if r['variant']==variant and r['metric']=='pressure_macro_f1'])
        gates[variant]={'count_simultaneous_ci_pass':bool(passing),'all_seed_mae_pass':all(r['delta']<0 for r in seed_checks),'scene_win_fraction':float(np.mean(scene_wins)),'pressure_mae_delta':float(pressure_mae),'pressure_macro_f1_delta':float(pressure_f1),'seed_checks':seed_checks,'pass':bool(passing and all(r['delta']<0 for r in seed_checks) and np.mean(scene_wins)>=.6 and pressure_mae<=0 and pressure_f1>=0)}
    atomic_json(outdir/'comparisons-and-ci.json',comparisons);write_csv(outdir/'comparisons-and-ci.csv',comparisons)
    write_csv(outdir/'paired-scene-deltas.csv',scene_deltas)
    atomic_json(outdir/'gate-decision.json',{'gates':gates,'decision':'Continue toward calibration and publication evidence' if gates['strfe_graph']['pass'] else 'Retain persistence; revise architecture/task before calibration or publication claims','strfe_only_alternative':gates['strfe']['pass'],'independent_test_sequences':scenes,'test_checkpoint_seal':sha(STUDY/'CHECKPOINTS_SEALED.json'),'pressure_semantics':'derived proxy, not emergency ground truth'})
