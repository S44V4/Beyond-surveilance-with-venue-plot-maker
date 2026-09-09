"""Protocol and immutable dataset inventory. No outcomes are used for splitting."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'outputs'/'evidence-20260907'
DATA=Path(r'C:\Users\abhij\Documents\Beyond Surveillance\data\raw\DroneCrowd')
PROTOCOL={
    'version':'evidence-v1', 'seeds':[23037,23038,23039], 'split_seed':20260907,
    'development_exposed_sequences':['001','011','050','083'],
    'test_sequences':22,'validation_sequences':22,
    'context_frames':8,'horizons':[5,15,30], 'temporal_unit':'normalized_annotation_frame_steps',
    'train_window_stride':5,'evaluation_window_stride':1,
    'strfe_epochs':40,'graph_epochs':40,'patience':8,'min_epochs':12,
    'batch_size':2,'learning_rate':0.0005,'weight_decay':0.0001,
    'feature_dim':64,'dropout':0.1,'gradient_clip':1.,
    'precision':'FP32 compute; frozen visual features stored float16',
    'feature_preprocessing':'valid-image-features-v1',
    'density_scale':100,'proxy_quantile':.95,'proxy_weights':{'density':1/3,'motion':1/3,'convergence':1/3},
    'pressure_classes':'training-split quartiles; ordinal derived pressure, not emergencies',
    'checkpoint_selection':'lowest validation mean of zone count MAE divided by fixed training count scale and pressure MAE; equal horizon and scene weights',
    'bootstrap_replicates':10000,'bootstrap_seed':20260907,
    'ci':'paired scene-cluster bootstrap; scenes weighted equally; training seeds evaluated separately and averaged without ensembling predictions',
    'primary_metrics':['zone_count_mae','zone_count_rmse','total_count_mae','total_count_rmse'],
    'gate':'For each learned variant: all 12 horizon/count-metric simultaneous one-sided 95% upper delta bounds <0 (Bonferroni across 24 variant/horizon/metric comparisons); every seed improves pooled zone and total MAE; at least 60% of test scenes improve both MAEs. Pressure must not regress in mean macro-F1 and must have nonpositive mean MAE delta. Otherwise keep persistence and revise.',
    'failure_analysis':'largest paired scene MAE regressions, largest total errors, low/high count and count-change strata, proxy confusion matrices; descriptive only, no retuning',
    'test_access':'final evaluation only after all six best checkpoints are sealed; repeat only to repair evaluation errors with an audit trail',
    'limitations':['DDPF original training corpus cannot be independently verified from supplied weights.','Different sequences may share physical locations; source-level scene independence cannot be guaranteed without location metadata.','Derived pressure measures future perception proxy agreement, not real hazards.','No calibrated physical time or causal intervention claims.']
}

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)

def freeze():
    STUDY.mkdir(parents=True,exist_ok=True)
    protocol_path=STUDY/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=PROTOCOL:raise ValueError('Frozen protocol differs: create a new study, never overwrite it')
    else:atomic_json(protocol_path,PROTOCOL)
    if (STUDY/'split-manifest.json').exists():return json.loads((STUDY/'split-manifest.json').read_text())
    groups={}; copies=0
    for path in sorted(DATA.rglob('img*.jpg')):
        sid=path.stem[3:6]
        if path.name in groups.setdefault(sid,{}): copies+=1
        else:groups[sid][path.name]=path
    exposed=set(PROTOCOL['development_exposed_sequences'])
    ordered=sorted(set(groups)-exposed,key=lambda s:hashlib.sha256(f"{PROTOCOL['split_seed']}:{s}".encode()).hexdigest())
    test=ordered[:22];val=ordered[22:44];train=sorted(set(groups)-set(test)-set(val))
    if len(test)!=22 or len(val)!=22:raise ValueError('Insufficient unseen sequences')
    partitions={'train':train,'val':val,'test':test}
    inventory={}; image_hash_splits={}
    for split,ids in partitions.items():
        for sid in ids:
            rows=[]
            for path in sorted(groups[sid].values()):
                annotation=path.parent.parent/'ground_truth'/f'GT_{path.stem}.mat'
                if not annotation.exists():raise ValueError(f'Missing {annotation}')
                digest=sha(path)
                if digest in image_hash_splits and image_hash_splits[digest]!=split:raise ValueError(f'Exact image duplicate across splits: {path}')
                image_hash_splits[digest]=split
                rows.append({'image':str(path.relative_to(DATA)),'annotation':str(annotation.relative_to(DATA)),'image_sha256':digest,'annotation_sha256':sha(annotation),'frame_index':int(path.stem[6:])})
            if len(rows)<38 or any(b['frame_index']!=a['frame_index']+1 for a,b in zip(rows,rows[1:])):raise ValueError(f'Incomplete/irregular sequence {sid}')
            inventory[sid]={'split':split,'frames':rows}
        print(f'Inventoried {split}: {len(ids)} complete sequences',flush=True)
    manifest={'protocol_sha256':sha(protocol_path),'ddpf_sha256':sha(ROOT/'models/balanced_ddpf.pt'),'dataset_root':str(DATA),'partitions':partitions,'sequences':inventory,'duplicate_local_copies_ignored':copies,'exact_image_overlap_across_splits':False,'all_available_sequences':len(groups),'all_unique_frames':sum(len(x) for x in groups.values())}
    atomic_json(STUDY/'split-manifest.json',manifest)
    atomic_json(STUDY/'environment.json',{'python':sys.version,'platform':platform.platform(),'packages':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()})
    return manifest

if __name__=='__main__':
    m=freeze();print(json.dumps({'sequences':{k:len(v) for k,v in m['partitions'].items()},'frames':m['all_unique_frames']}))
