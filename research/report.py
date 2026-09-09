"""Render the saved evidence without rerunning inference or modifying the decision."""
import json
from pathlib import Path
import numpy as np
from research.protocol import STUDY,PROTOCOL,atomic_json,sha

def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=STUDY/'evaluation'
    decision=json.loads((folder/'gate-decision.json').read_text());comparisons=json.loads((folder/'comparisons-and-ci.json').read_text());rows=json.loads((folder/'scene-horizon-metrics.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,metric in zip(axes.flat,PROTOCOL['primary_metrics']):
        for variant,color,shift in [('strfe','#2456d6',-.08),('strfe_graph','#a55b10',.08)]:
            subset=[r for r in comparisons if r['variant']==variant and r['metric']==metric]
            x=np.arange(3)+shift;y=np.array([r['delta'] for r in subset]);lo=np.array([r['delta_ci95'][0] for r in subset]);hi=np.array([r['delta_ci95'][1] for r in subset])
            ax.errorbar(x,y,yerr=np.stack([np.maximum(y-lo,0),np.maximum(hi-y,0)]),fmt='o',capsize=4,label=variant,color=color)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(3),PROTOCOL['horizons']);ax.set_xlabel('Horizon (annotation frame steps)');ax.set_ylabel('Learned − persistence (people)');ax.set_title(metric.replace('_',' '));ax.legend()
    fig.suptitle('Paired sequence-cluster differences, marginal 95% intervals\nNegative count differences favor the learned model')
    fig.savefig(folder/'count-comparison.png',dpi=180);plt.close(fig)
    lines=['# Independent held-out model-evidence report','',f"**Decision: {decision['decision']}.**",'','This is a fixed sequence-disjoint holdout study, not cross-validation. All three training seeds were selected using validation only. Test predictions were generated after checkpoint sealing.','', '## Dataset and protocol','', '68 training, 22 validation and 22 test sequences; all 33,600 unique annotated frames. The previously inspected development sequences are excluded from test. Training uses stride-five temporal windows, eight-frame context, and 5/15/30 annotation-step horizons. Validation and test use every eligible window. Each model has a 40-epoch budget, a minimum of 12 epochs and eight-epoch patience.','', 'Counts use manual future head annotations. Persistence receives only current DDPF counts and current perception-derived pressure. Pressure targets and class cutoffs use training-only calibration; these are not emergency labels.','', 'The dataset authors report 112 clips from 70 scenarios, so clip separation does not by itself establish separation of physical locations. [DroneCrowd dataset](https://github.com/VisDrone/DroneCrowd). Supplied DDPF training-set provenance cannot be independently proven from weights.','', '## Count and pressure comparisons','', '| Model | Horizon | Metric | Persistence | Learned | Difference [95% CI] |','|---|---:|---|---:|---:|---:|']
    for row in comparisons:
        lo,hi=row['delta_ci95'];lines.append(f"| {row['variant']} | {row['horizon']} | {row['metric']} | {row['persistence']:.4f} | {row['learned']:.4f} | {row['delta']:.4f} [{lo:.4f}, {hi:.4f}] |")
    lines.extend(['','## Decision rule and uncertainty','','The protocol was frozen before full training. The count gate uses simultaneous one-sided 95% upper bounds, Bonferroni-adjusted over 24 variant/horizon/metric comparisons; the table shows marginal two-sided 95% intervals. We resample whole test sequences 10,000 times, keeping paired models, all overlapping windows and all training seeds together. The point estimate weights sequences equally. RMSE is the square root of the mean sequence MSE. Seed metrics are averaged; predictions are not ensembled. These intervals are conditional on the fixed split and proxy definition, and assume independence between sequences.',''])
    for variant,result in decision['gates'].items():lines.append(f"- **{variant}: {'PASS' if result['pass'] else 'FAIL'}**. Simultaneous count gate: {result['count_simultaneous_ci_pass']}; all-seed MAE improvement: {result['all_seed_mae_pass']}; scenes improving both count MAEs: {result['scene_win_fraction']:.1%}; pressure MAE delta {result['pressure_mae_delta']:.4f}; pressure macro-F1 delta {result['pressure_macro_f1_delta']:.4f}.")
    lookup={(r['seed'],r['scene'],r['horizon'],r['variant']):r for r in rows}
    regressions=[]
    for scene in decision['independent_test_sequences']:
        delta=np.mean([lookup[(seed,scene,h,'strfe_graph')]['total_count_mae']-lookup[(seed,scene,h,'persistence')]['total_count_mae'] for seed in PROTOCOL['seeds'] for h in PROTOCOL['horizons']])
        regressions.append((float(delta),scene))
    lines.extend(['','## Failure cases','','Largest scene-level STRFE+Graph regressions, averaged over horizons and training seeds:','','| Sequence | Total-count MAE difference |','|---|---:|'])
    for delta,scene in sorted(regressions,reverse=True)[:10]:lines.append(f'| {scene} | {delta:.3f} |')
    lines.extend(['','`failure-cases.csv` identifies the worst timestamp/horizon predictions. `strata.csv` separates count levels and count changes. `scene-horizon-metrics.json` includes proxy confusion matrices, class supports, precision, recall and F1. These analyses are descriptive; no model was retuned using them.','','## Reproduction and artifacts','','Run `python -m research.run` from the project root. The runner uses the frozen protocol and split manifest, resumes complete cache/epoch checkpoints and verifies source and model hashes. Source snapshots, environment versions, all best/last checkpoints, per-epoch history, pressure calibration, per-sequence prediction arrays, CSV/JSON metrics, confidence intervals and this report are retained in this study directory.','','No calibration, publication-readiness, causal intervention, MAPPO or deployment claim follows from merely completing this experiment. If the graph model fails this gate, persistence remains the accepted baseline for the current task.'])
    (STUDY/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    artifacts={str(path.relative_to(STUDY)):sha(path) for path in folder.iterdir() if path.is_file()}
    atomic_json(STUDY/'EVALUATION_COMPLETE.json',{'decision':decision['decision'],'gate_pass':decision['gates']['strfe_graph']['pass'],'report_sha256':sha(STUDY/'REPORT.md'),'artifacts':artifacts})
